/**
 * CrewAI & Agent Frameworks Section
 * Module 7: Multi-Agent Systems & Orchestration
 */

export const crewaiagentframeworksSection = {
  id: 'crewai-agent-frameworks',
  title: 'CrewAI & Agent Frameworks',
  content: `# CrewAI & Agent Frameworks

Master building multi-agent systems using CrewAI and other popular frameworks.

## Overview: Why Use Frameworks?

Agent frameworks provide:

- **Abstractions**: Less boilerplate code
- **Patterns**: Proven collaboration patterns
- **Tools**: Built-in utilities
- **Testing**: Testing utilities
- **Examples**: Learn from examples

### Popular Frameworks

**CrewAI**: Role-based crews  
**LangGraph**: Graph-based workflows  
**AutoGPT**: Autonomous task completion  
**MetaGPT**: Software development teams  
**AgentOps**: Observability for agents  

## CrewAI Fundamentals

CrewAI models agent teams as "crews" with roles:

\`\`\`python
from crewai import Agent, Task, Crew, Process

# 1. Define Agents with Roles
researcher = Agent(
    role='Senior Researcher',
    goal='Conduct comprehensive research on {topic}',
    backstory="""You are an expert researcher with years of experience 
    in gathering and analyzing information from various sources.""",
    verbose=True,
    allow_delegation=False
)

writer = Agent(
    role='Tech Writer',
    goal='Write engaging and informative articles about {topic}',
    backstory="""You are a skilled writer who excels at making complex 
    topics accessible to general audiences.""",
    verbose=True,
    allow_delegation=False
)

editor = Agent(
    role='Editor',
    goal='Review and improve content quality',
    backstory="""You are a meticulous editor with an eye for detail 
    and a commitment to excellence.""",
    verbose=True,
    allow_delegation=False
)

# 2. Define Tasks
research_task = Task(
    description="""Research the following topic: {topic}
    
    Requirements:
    - Gather key facts and concepts
    - Find recent developments
    - Identify expert opinions
    """,
    agent=researcher,
    expected_output="Comprehensive research document"
)

writing_task = Task(
    description="""Using the research provided, write a clear and 
    engaging article about {topic}.
    
    Requirements:
    - 500-700 words
    - Accessible language
    - Include key findings
    """,
    agent=writer,
    expected_output="Draft article"
)

editing_task = Task(
    description="""Review and improve the article draft.
    
    Focus on:
    - Clarity and flow
    - Grammar and style
    - Factual accuracy
    """,
    agent=editor,
    expected_output="Final polished article"
)

# 3. Create Crew
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,  # Tasks run in order
    verbose=True
)

# 4. Execute
result = crew.kickoff (inputs={"topic": "quantum computing"})
print(result)
\`\`\`

## CrewAI with Custom Tools

\`\`\`python
from crewai_tools import tool
import requests

@tool("Web Search")
def web_search (query: str) -> str:
    """Search the web for information."""
    # Simplified - use real search API
    return f"Search results for: {query}"

@tool("File Reader")
def read_file (filepath: str) -> str:
    """Read content from a file."""
    try:
        with open (filepath, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

@tool("Calculator")
def calculate (expression: str) -> float:
    """Evaluate mathematical expressions."""
    try:
        return eval (expression)
    except Exception as e:
        return f"Error: {e}"

# Agent with tools
analyst = Agent(
    role='Data Analyst',
    goal='Analyze data and provide insights',
    backstory='Expert data analyst with strong analytical skills',
    tools=[web_search, read_file, calculate],
    verbose=True
)

analysis_task = Task(
    description="""Analyze the data and provide key insights.
    Use available tools as needed.""",
    agent=analyst,
    expected_output="Analysis report with insights"
)

crew = Crew(
    agents=[analyst],
    tasks=[analysis_task],
    verbose=True
)

result = crew.kickoff()
\`\`\`

## Hierarchical Process in CrewAI

Manager delegates to workers:

\`\`\`python
from crewai import Crew, Process

# Worker agents
code_agent = Agent(
    role='Software Developer',
    goal='Write clean, efficient code',
    backstory='Experienced developer proficient in Python',
    verbose=True
)

test_agent = Agent(
    role='QA Engineer',
    goal='Ensure code quality through testing',
    backstory='Thorough QA engineer who catches all bugs',
    verbose=True
)

doc_agent = Agent(
    role='Technical Writer',
    goal='Create clear documentation',
    backstory='Technical writer who makes complex topics simple',
    verbose=True
)

# Tasks (manager will delegate these)
tasks = [
    Task(
        description="Implement a function to sort a list",
        expected_output="Working code implementation"
    ),
    Task(
        description="Write tests for the sorting function",
        expected_output="Comprehensive test suite"
    ),
    Task(
        description="Document the sorting function",
        expected_output="Clear documentation"
    )
]

# Hierarchical crew (CrewAI creates manager automatically)
crew = Crew(
    agents=[code_agent, test_agent, doc_agent],
    tasks=tasks,
    process=Process.hierarchical,  # Manager delegates
    verbose=True,
    manager_llm="gpt-4"  # LLM for manager
)

result = crew.kickoff()
\`\`\`

## Memory and Context in CrewAI

\`\`\`python
from crewai import Agent, Task, Crew
from crewai.memory import ShortTermMemory, LongTermMemory, EntityMemory

# Agent with memory
remembering_agent = Agent(
    role='Personal Assistant',
    goal='Help user with tasks while remembering context',
    backstory='Attentive assistant who remembers user preferences',
    memory=True,  # Enable memory
    verbose=True
)

# Crew with memory systems
crew = Crew(
    agents=[remembering_agent],
    tasks=[
        Task(
            description="User says: I prefer dark mode",
            expected_output="Acknowledgment"
        ),
        Task(
            description="What are my preferences?",
            expected_output="Remembered preferences"
        )
    ],
    memory=True,  # Crew-wide memory
    verbose=True
)

result = crew.kickoff()
# Agent will remember "dark mode" preference
\`\`\`

## AutoGPT Pattern

Autonomous goal-driven agent:

\`\`\`python
class AutoGPTAgent:
    """Agent that autonomously breaks down and completes goals."""
    
    def __init__(self, goal: str, tools: List[callable]):
        self.goal = goal
        self.tools = tools
        self.completed_tasks = []
        self.max_iterations = 10
    
    async def run (self) -> Dict[str, Any]:
        """Run agent autonomously."""
        print(f"Goal: {self.goal}\\n")
        
        for iteration in range (self.max_iterations):
            print(f"Iteration {iteration + 1}/{self.max_iterations}")
            
            # 1. Think: What should I do next?
            next_action = await self._think()
            
            if next_action == "GOAL_COMPLETE":
                print("✅ Goal achieved!")
                return {
                    "success": True,
                    "completed_tasks": self.completed_tasks
                }
            
            # 2. Act: Execute the action
            result = await self._act (next_action)
            
            # 3. Observe: Record result
            self.completed_tasks.append({
                "action": next_action,
                "result": result
            })
            
            print(f"  Completed: {next_action}\\n")
        
        return {
            "success": False,
            "message": "Max iterations reached",
            "completed_tasks": self.completed_tasks
        }
    
    async def _think (self) -> str:
        """Decide next action."""
        context = self._build_context()
        
        prompt = f"""Goal: {self.goal}

Context:
{context}

Available actions:
{self._list_available_actions()}

What should be the next action to achieve the goal?
Reply with action name or "GOAL_COMPLETE" if done."""
        
        # Use LLM to decide
        response = await self._call_llm (prompt)
        return response.strip()
    
    async def _act (self, action: str) -> Any:
        """Execute action using tools."""
        # Find matching tool
        for tool in self.tools:
            if tool.__name__ == action:
                return await tool()
        
        return f"Action {action} not found"
    
    def _build_context (self) -> str:
        """Build context from completed tasks."""
        if not self.completed_tasks:
            return "No tasks completed yet"
        
        context = "Completed tasks:\\n"
        for task in self.completed_tasks[-5:]:  # Last 5
            context += f"- {task['action']}: {task['result']}\\n"
        return context
    
    def _list_available_actions (self) -> str:
        """List available tools."""
        return ", ".join([tool.__name__ for tool in self.tools])
    
    async def _call_llm (self, prompt: str) -> str:
        """Call LLM."""
        # Implement LLM call
        pass

# Usage
async def research():
    return "Research completed: Found key facts"

async def write():
    return "Article written based on research"

async def review():
    return "Review complete: Article approved"

agent = AutoGPTAgent(
    goal="Write and publish an article about AI",
    tools=[research, write, review]
)

result = await agent.run()
\`\`\`

## MetaGPT Pattern

Software team simulation:

\`\`\`python
class MetaGPTTeam:
    """Simulate software development team."""
    
    def __init__(self):
        # Define roles
        self.product_manager = Agent(
            role="Product Manager",
            goal="Define requirements and user stories"
        )
        
        self.architect = Agent(
            role="Architect",
            goal="Design system architecture"
        )
        
        self.engineer = Agent(
            role="Engineer",
            goal="Implement features"
        )
        
        self.qa = Agent(
            role="QA Engineer",
            goal="Test and validate"
        )
    
    async def build_product (self, idea: str) -> Dict[str, Any]:
        """Go from idea to implemented product."""
        
        # 1. PM creates requirements
        requirements = await self.product_manager.execute(
            f"Create detailed requirements for: {idea}"
        )
        
        # 2. Architect designs system
        architecture = await self.architect.execute(
            f"Design architecture for: {requirements}"
        )
        
        # 3. Engineer implements
        code = await self.engineer.execute(
            f"Implement based on: {architecture}"
        )
        
        # 4. QA tests
        test_results = await self.qa.execute(
            f"Test this implementation: {code}"
        )
        
        return {
            "requirements": requirements,
            "architecture": architecture,
            "code": code,
            "tests": test_results
        }

# Usage
team = MetaGPTTeam()
result = await team.build_product("Task management app")
\`\`\`

## AgentOps for Monitoring

\`\`\`python
import agentops

# Initialize AgentOps
agentops.init (api_key="your_key")

class MonitoredAgent:
    """Agent with monitoring."""
    
    def __init__(self, name: str):
        self.name = name
        self.session = agentops.start_session()
    
    @agentops.record_action
    async def execute (self, task: str) -> str:
        """Execute task with monitoring."""
        # This call is automatically logged
        result = await self._do_work (task)
        
        # Log custom events
        agentops.record_event(
            event_type="task_completed",
            properties={
                "agent": self.name,
                "task": task,
                "result_length": len (result)
            }
        )
        
        return result
    
    async def _do_work (self, task: str) -> str:
        """Actual work."""
        import asyncio
        await asyncio.sleep(1)
        return f"Completed: {task}"
    
    def end_session (self):
        """End monitoring session."""
        self.session.end_session("Success")

# Usage
agent = MonitoredAgent("WorkerAgent")
result = await agent.execute("Process data")
agent.end_session()

# View metrics in AgentOps dashboard
\`\`\`

## Framework Comparison

\`\`\`python
def choose_framework (requirements: Dict[str, bool]) -> str:
    """Choose appropriate framework."""
    
    if requirements.get("role_based_teams"):
        return "CrewAI - Best for role-based agent teams"
    
    if requirements.get("complex_workflows"):
        return "LangGraph - Best for complex stateful workflows"
    
    if requirements.get("autonomous_agents"):
        return "AutoGPT - Best for autonomous task completion"
    
    if requirements.get("software_teams"):
        return "MetaGPT - Best for simulating dev teams"
    
    if requirements.get("monitoring_focus"):
        return "AgentOps - Best for monitoring and observability"
    
    return "Start with CrewAI (easiest to learn)"

# Example
requirements = {
    "role_based_teams": True,
    "complex_workflows": False
}

framework = choose_framework (requirements)
print(f"Recommended: {framework}")
\`\`\`

## Custom Framework

Build your own lightweight framework:

\`\`\`python
from dataclasses import dataclass
from typing import List, Callable, Any

@dataclass
class AgentRole:
    """Define an agent role."""
    name: str
    description: str
    capabilities: List[str]
    llm_config: Dict[str, Any]

@dataclass
class AgentTask:
    """Task to be executed."""
    description: str
    assigned_to: str
    dependencies: List[str]
    output_key: str

class SimpleFramework:
    """Minimal multi-agent framework."""
    
    def __init__(self):
        self.roles: Dict[str, AgentRole] = {}
        self.context: Dict[str, Any] = {}
    
    def register_role (self, role: AgentRole):
        """Register an agent role."""
        self.roles[role.name] = role
    
    async def execute_tasks(
        self,
        tasks: List[AgentTask]
    ) -> Dict[str, Any]:
        """Execute tasks in dependency order."""
        completed = set()
        results = {}
        
        while len (completed) < len (tasks):
            # Find ready tasks
            ready = [
                t for t in tasks
                if t.output_key not in completed
                and all (dep in completed for dep in t.dependencies)
            ]
            
            if not ready:
                raise ValueError("Circular dependency or stuck")
            
            # Execute ready tasks
            for task in ready:
                result = await self._execute_task (task)
                results[task.output_key] = result
                completed.add (task.output_key)
                self.context[task.output_key] = result
        
        return results
    
    async def _execute_task (self, task: AgentTask) -> Any:
        """Execute single task."""
        role = self.roles.get (task.assigned_to)
        if not role:
            raise ValueError (f"Role {task.assigned_to} not found")
        
        print(f"[{role.name}] {task.description}")
        
        # Build context from dependencies
        context = {
            dep: self.context[dep]
            for dep in task.dependencies
            if dep in self.context
        }
        
        # Execute with LLM
        result = await self._call_llm(
            role,
            task.description,
            context
        )
        
        return result
    
    async def _call_llm(
        self,
        role: AgentRole,
        task: str,
        context: Dict
    ) -> str:
        """Call LLM for role."""
        # Implement actual LLM call
        return f"Result from {role.name}"

# Usage
framework = SimpleFramework()

# Register roles
framework.register_role(AgentRole(
    name="researcher",
    description="Conducts research",
    capabilities=["web_search", "analysis"],
    llm_config={"model": "gpt-4"}
))

framework.register_role(AgentRole(
    name="writer",
    description="Writes content",
    capabilities=["writing"],
    llm_config={"model": "gpt-4"}
))

# Define tasks
tasks = [
    AgentTask(
        description="Research topic",
        assigned_to="researcher",
        dependencies=[],
        output_key="research"
    ),
    AgentTask(
        description="Write article",
        assigned_to="writer",
        dependencies=["research"],
        output_key="article"
    )
]

# Execute
results = await framework.execute_tasks (tasks)
\`\`\`

## Best Practices

1. **Start Simple**: Use existing frameworks first
2. **Understand Patterns**: Learn framework patterns
3. **Customize When Needed**: Extend, don't rewrite
4. **Monitor Everything**: Use observability tools
5. **Version Control**: Track framework versions
6. **Test Thoroughly**: Unit test agent behaviors
7. **Document**: Document your agent setup

## Next Steps

You now understand agent frameworks. Next, learn:
- Debugging multi-agent systems
- Human-in-the-loop patterns
- Production deployment
`,
};
