/**
 * Task Decomposition & Planning Section
 * Module 7: Multi-Agent Systems & Orchestration
 */

export const taskdecompositionplanningSection = {
  id: 'task-decomposition-planning',
  title: 'Task Decomposition & Planning',
  content: `# Task Decomposition & Planning

Master breaking down complex tasks into agent-executable subtasks and creating effective execution plans.

## Overview: Why Decomposition Matters

Complex tasks are too large for single agents. Decomposition enables:

- **Parallel Execution**: Multiple agents work simultaneously
- **Specialization**: Route subtasks to expert agents
- **Progress Tracking**: Monitor completion of each subtask
- **Error Isolation**: Failures don't cascade to entire system
- **Incremental Results**: Get partial results as you go

### Decomposition Challenges

- **Identifying Dependencies**: What must happen first?
- **Appropriate Granularity**: Too big or too small?
- **Resource Allocation**: Which agent for which task?
- **Dynamic Adaptation**: Plans change as you learn

## Hierarchical Task Decomposition

Break tasks into tree structure:

\`\`\`python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import openai

class TaskStatus(Enum):
    """Status of a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

@dataclass
class Task:
    """A task in the hierarchy."""
    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    parent_id: Optional[str] = None
    subtasks: List['Task'] = field (default_factory=list)
    assigned_to: Optional[str] = None
    dependencies: List[str] = field (default_factory=list)
    result: Optional[Any] = None
    
    def add_subtask (self, subtask: 'Task'):
        """Add subtask to this task."""
        subtask.parent_id = self.id
        self.subtasks.append (subtask)
    
    def is_leaf (self) -> bool:
        """Check if this is a leaf task (no subtasks)."""
        return len (self.subtasks) == 0
    
    def can_execute (self, completed_tasks: set[str]) -> bool:
        """Check if task can be executed (dependencies met)."""
        return all (dep in completed_tasks for dep in self.dependencies)

class TaskDecomposer:
    """Decomposes complex tasks hierarchically."""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.task_counter = 0
    
    async def decompose(
        self,
        goal: str,
        max_depth: int = 3,
        current_depth: int = 0
    ) -> Task:
        """Recursively decompose task."""
        # Create root task
        root_task = Task(
            id=self._generate_id(),
            description=goal,
            status=TaskStatus.PENDING
        )
        
        if current_depth >= max_depth:
            return root_task
        
        # Break down into subtasks
        subtask_descriptions = await self._get_subtasks (goal)
        
        for desc in subtask_descriptions:
            # Create subtask
            subtask = Task(
                id=self._generate_id(),
                description=desc,
                status=TaskStatus.PENDING
            )
            
            # Recursively decompose if needed
            if self._needs_decomposition (desc) and current_depth < max_depth - 1:
                subtask = await self.decompose (desc, max_depth, current_depth + 1)
            
            root_task.add_subtask (subtask)
        
        return root_task
    
    async def _get_subtasks (self, task: str) -> List[str]:
        """Get subtasks for a task using LLM."""
        prompt = f"""Break this task into 3-5 specific subtasks:

Task: {task}

Requirements:
- Each subtask should be concrete and actionable
- Subtasks should cover the full task
- Keep subtasks at similar granularity

Format: One subtask per line, starting with "- "
"""
        
        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a task planning expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        # Parse subtasks
        text = response.choices[0].message.content
        subtasks = []
        for line in text.split("\\n"):
            line = line.strip()
            if line.startswith("-"):
                subtasks.append (line[1:].strip())
            elif line and not line.endswith(":"):
                subtasks.append (line)
        
        return subtasks
    
    def _needs_decomposition (self, task: str) -> bool:
        """Heuristic to determine if task needs further decomposition."""
        # Tasks with certain keywords likely need decomposition
        complex_keywords = ["build", "create", "implement", "design", "develop"]
        return any (keyword in task.lower() for keyword in complex_keywords)
    
    def _generate_id (self) -> str:
        """Generate unique task ID."""
        self.task_counter += 1
        return f"task_{self.task_counter}"
    
    def visualize_tree (self, task: Task, indent: int = 0) -> str:
        """Visualize task tree."""
        lines = []
        prefix = "  " * indent
        status_icon = {
            TaskStatus.PENDING: "⭕",
            TaskStatus.IN_PROGRESS: "🔄",
            TaskStatus.COMPLETED: "✅",
            TaskStatus.FAILED: "❌",
            TaskStatus.BLOCKED: "🚫"
        }
        
        icon = status_icon.get (task.status, "⭕")
        lines.append (f"{prefix}{icon} {task.description} [{task.id}]")
        
        for subtask in task.subtasks:
            lines.append (self.visualize_tree (subtask, indent + 1))
        
        return "\\n".join (lines)

# Usage
decomposer = TaskDecomposer()

# Decompose complex goal
root_task = await decomposer.decompose(
    "Build a web application for task management",
    max_depth=3
)

print(decomposer.visualize_tree (root_task))
# ⭕ Build a web application for task management [task_1]
#   ⭕ Design database schema [task_2]
#   ⭕ Create backend API [task_3]
#     ⭕ Implement authentication [task_4]
#     ⭕ Create CRUD endpoints [task_5]
#   ⭕ Build frontend interface [task_6]
\`\`\`

## Dependency-Aware Planning

Tasks often have dependencies - some must complete before others:

\`\`\`python
from typing import Set

class DependencyPlanner:
    """Plans task execution considering dependencies."""
    
    def __init__(self):
        pass
    
    async def create_plan_with_dependencies(
        self,
        tasks: List[Task]
    ) -> List[List[Task]]:
        """Create execution plan grouped by dependencies.
        
        Returns list of task groups that can be executed in parallel.
        """
        # Analyze dependencies
        tasks_with_deps = await self._analyze_dependencies (tasks)
        
        # Create execution stages
        stages = self._create_execution_stages (tasks_with_deps)
        
        return stages
    
    async def _analyze_dependencies(
        self,
        tasks: List[Task]
    ) -> List[Task]:
        """Analyze and set dependencies between tasks."""
        # Use LLM to identify dependencies
        task_descriptions = "\\n".join([
            f"{i+1}. {task.description}"
            for i, task in enumerate (tasks)
        ])
        
        prompt = f"""Analyze dependencies between these tasks:

{task_descriptions}

For each task, list which tasks must be completed BEFORE it can start.

Format:
Task X depends on: [list of task numbers, or "None"]
"""
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You analyze task dependencies."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        # Parse dependencies
        dep_text = response.choices[0].message.content
        
        import re
        for line in dep_text.split("\\n"):
            # Extract task number and dependencies
            match = re.search (r'Task (\\d+) depends on: (.+)', line)
            if match:
                task_num = int (match.group(1)) - 1
                dep_text = match.group(2).strip()
                
                if task_num < len (tasks):
                    if dep_text.lower() != "none":
                        # Extract dependency numbers
                        dep_nums = [int (d) - 1 for d in re.findall (r'\\d+', dep_text)]
                        tasks[task_num].dependencies = [
                            tasks[i].id for i in dep_nums if i < len (tasks)
                        ]
        
        return tasks
    
    def _create_execution_stages(
        self,
        tasks: List[Task]
    ) -> List[List[Task]]:
        """Create stages of tasks that can execute in parallel."""
        stages = []
        remaining = set (task.id for task in tasks)
        completed = set()
        task_dict = {task.id: task for task in tasks}
        
        while remaining:
            # Find tasks that can execute now
            current_stage = []
            
            for task_id in list (remaining):
                task = task_dict[task_id]
                if task.can_execute (completed):
                    current_stage.append (task)
                    remaining.remove (task_id)
            
            if not current_stage:
                # Circular dependency or error
                raise ValueError("Circular dependency detected or blocked tasks")
            
            stages.append (current_stage)
            
            # Mark current stage as completed
            completed.update (task.id for task in current_stage)
        
        return stages
    
    def visualize_plan (self, stages: List[List[Task]]) -> str:
        """Visualize execution plan."""
        output = ["=== EXECUTION PLAN ===" ""]
        
        for i, stage in enumerate (stages, 1):
            output.append (f"Stage {i} (Parallel):")
            for task in stage:
                deps = ", ".join (task.dependencies) if task.dependencies else "None"
                output.append (f"  - {task.description}")
                output.append (f"    Dependencies: {deps}")
                output.append (f"    Assigned to: {task.assigned_to or 'TBD'}")
            output.append("")
        
        return "\\n".join (output)

# Usage
planner = DependencyPlanner()

# Create tasks
tasks = [
    Task (id="t1", description="Design database schema"),
    Task (id="t2", description="Set up development environment"),
    Task (id="t3", description="Implement user authentication"),
    Task (id="t4", description="Create CRUD API endpoints"),
    Task (id="t5", description="Build frontend UI"),
    Task (id="t6", description="Write integration tests"),
]

# Analyze and plan
stages = await planner.create_plan_with_dependencies (tasks)

print(planner.visualize_plan (stages))
# Stage 1 (Parallel):
#   - Design database schema
#   - Set up development environment
#
# Stage 2 (Parallel):
#   - Implement user authentication
#
# Stage 3 (Parallel):
#   - Create CRUD API endpoints
#
# Stage 4 (Parallel):
#   - Build frontend UI
#   - Write integration tests
\`\`\`

## Dynamic Replanning

Plans need to adapt as tasks complete or fail:

\`\`\`python
class DynamicPlanner:
    """Plans and replans based on execution results."""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.execution_history: List[Dict[str, Any]] = []
    
    async def replan(
        self,
        original_goal: str,
        completed_tasks: List[Task],
        failed_tasks: List[Task],
        remaining_tasks: List[Task]
    ) -> List[Task]:
        """Replan based on current state."""
        # Analyze what worked and what didn't
        context = self._build_context(
            original_goal,
            completed_tasks,
            failed_tasks,
            remaining_tasks
        )
        
        # Generate new plan
        new_tasks = await self._generate_revised_plan (context)
        
        return new_tasks
    
    def _build_context(
        self,
        goal: str,
        completed: List[Task],
        failed: List[Task],
        remaining: List[Task]
    ) -> str:
        """Build context for replanning."""
        context = f"""Original Goal: {goal}

Completed Tasks:
{self._format_tasks (completed)}

Failed Tasks:
{self._format_tasks (failed)}

Remaining Tasks:
{self._format_tasks (remaining)}
"""
        return context
    
    def _format_tasks (self, tasks: List[Task]) -> str:
        """Format tasks for display."""
        if not tasks:
            return "- None"
        return "\\n".join([f"- {task.description}" for task in tasks])
    
    async def _generate_revised_plan (self, context: str) -> List[Task]:
        """Generate revised plan using LLM."""
        prompt = f"""{context}

Based on progress so far, create revised plan to achieve the goal.

Consider:
- What we've learned from completed tasks
- Why failed tasks failed
- More efficient approach for remaining work

Output new task list in order."""
        
        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an adaptive planner."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        # Parse into tasks
        text = response.choices[0].message.content
        tasks = []
        task_id = 1
        
        for line in text.split("\\n"):
            line = line.strip()
            if line and (line.startswith("-") or line[0].isdigit()):
                # Clean up line
                desc = line.lstrip("-0123456789. ")
                if desc:
                    tasks.append(Task(
                        id=f"task_{task_id}",
                        description=desc
                    ))
                    task_id += 1
        
        return tasks

# Usage
replanner = DynamicPlanner()

# Some tasks completed, one failed
completed = [
    Task (id="t1", description="Research requirements", status=TaskStatus.COMPLETED),
    Task (id="t2", description="Design architecture", status=TaskStatus.COMPLETED)
]

failed = [
    Task (id="t3", description="Implement feature X", status=TaskStatus.FAILED)
]

remaining = [
    Task (id="t4", description="Implement feature Y"),
    Task (id="t5", description="Write tests")
]

# Replan
new_plan = await replanner.replan(
    original_goal="Build web application",
    completed_tasks=completed,
    failed_tasks=failed,
    remaining_tasks=remaining
)

print("Revised Plan:")
for task in new_plan:
    print(f"- {task.description}")
\`\`\`

## Resource-Aware Planning

Consider agent availability and capacity:

\`\`\`python
@dataclass
class Agent:
    """Agent with capacity tracking."""
    name: str
    capabilities: List[str]
    max_concurrent_tasks: int
    current_tasks: List[str] = field (default_factory=list)
    
    def can_handle (self, task: Task) -> bool:
        """Check if agent can handle this task."""
        # Check capacity
        if len (self.current_tasks) >= self.max_concurrent_tasks:
            return False
        
        # Check capabilities (simple keyword matching)
        task_lower = task.description.lower()
        return any (cap.lower() in task_lower for cap in self.capabilities)
    
    def assign_task (self, task_id: str):
        """Assign task to agent."""
        if len (self.current_tasks) < self.max_concurrent_tasks:
            self.current_tasks.append (task_id)
    
    def complete_task (self, task_id: str):
        """Mark task as complete."""
        if task_id in self.current_tasks:
            self.current_tasks.remove (task_id)

class ResourcePlanner:
    """Plans considering agent resources."""
    
    def __init__(self, agents: List[Agent]):
        self.agents = agents
    
    def assign_tasks(
        self,
        tasks: List[Task]
    ) -> Dict[str, List[Task]]:
        """Assign tasks to agents considering resources."""
        assignments: Dict[str, List[Task]] = {agent.name: [] for agent in self.agents}
        unassigned = []
        
        for task in tasks:
            # Find best agent
            best_agent = self._find_best_agent (task)
            
            if best_agent:
                best_agent.assign_task (task.id)
                task.assigned_to = best_agent.name
                assignments[best_agent.name].append (task)
            else:
                unassigned.append (task)
        
        if unassigned:
            print(f"Warning: {len (unassigned)} tasks unassigned")
        
        return assignments
    
    def _find_best_agent (self, task: Task) -> Optional[Agent]:
        """Find best available agent for task."""
        # Filter to agents that can handle it
        capable = [agent for agent in self.agents if agent.can_handle (task)]
        
        if not capable:
            return None
        
        # Choose least loaded agent
        return min (capable, key=lambda a: len (a.current_tasks))
    
    def get_workload_distribution (self) -> Dict[str, int]:
        """Get current workload per agent."""
        return {
            agent.name: len (agent.current_tasks)
            for agent in self.agents
        }

# Usage
agents = [
    Agent("Researcher", ["research", "analyze"], max_concurrent_tasks=3),
    Agent("Coder", ["code", "implement", "build"], max_concurrent_tasks=2),
    Agent("Tester", ["test", "validate"], max_concurrent_tasks=4)
]

resource_planner = ResourcePlanner (agents)

tasks = [
    Task("t1", "Research market trends"),
    Task("t2", "Implement login feature"),
    Task("t3", "Test user registration"),
    Task("t4", "Analyze competitor products"),
    Task("t5", "Build API endpoints"),
]

assignments = resource_planner.assign_tasks (tasks)

for agent_name, agent_tasks in assignments.items():
    print(f"{agent_name}:")
    for task in agent_tasks:
        print(f"  - {task.description}")
\`\`\`

## Adaptive Granularity

Adjust task granularity based on context:

\`\`\`python
class AdaptiveDecomposer:
    """Decomposes with adaptive granularity."""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
    
    async def decompose_adaptive(
        self,
        task: str,
        agent_expertise: str = "intermediate",
        time_available: str = "normal"
    ) -> List[Task]:
        """Decompose with appropriate granularity."""
        # Determine target granularity
        granularity = self._determine_granularity (agent_expertise, time_available)
        
        prompt = f"""Break down this task with {granularity} granularity:

Task: {task}

Agent expertise: {agent_expertise}
Time available: {time_available}

{self._get_granularity_instructions (granularity)}

Output: List of subtasks, one per line.
"""
        
        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a task planner."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        # Parse tasks
        tasks = []
        task_id = 1
        
        for line in response.choices[0].message.content.split("\\n"):
            line = line.strip()
            if line and (line.startswith("-") or line[0].isdigit()):
                desc = line.lstrip("-0123456789. ")
                if desc:
                    tasks.append(Task (f"task_{task_id}", desc))
                    task_id += 1
        
        return tasks
    
    def _determine_granularity(
        self,
        expertise: str,
        time: str
    ) -> str:
        """Determine appropriate granularity."""
        if expertise == "beginner":
            return "fine"  # More detailed steps
        elif expertise == "expert":
            return "coarse"  # High-level steps
        elif time == "urgent":
            return "coarse"  # Fewer, bigger tasks
        else:
            return "medium"
    
    def _get_granularity_instructions (self, granularity: str) -> str:
        """Get instructions for granularity level."""
        instructions = {
            "fine": "Break into very specific, detailed steps. Each step should take ~5-10 minutes.",
            "medium": "Break into clear, actionable steps. Each step should take ~30-60 minutes.",
            "coarse": "Break into high-level steps. Each step should take ~2-4 hours."
        }
        return instructions.get (granularity, instructions["medium"])

# Usage
adaptive = AdaptiveDecomposer()

# For beginner agent - fine-grained
beginner_tasks = await adaptive.decompose_adaptive(
    "Implement user authentication",
    agent_expertise="beginner",
    time_available="normal"
)

# For expert agent - coarse-grained
expert_tasks = await adaptive.decompose_adaptive(
    "Implement user authentication",
    agent_expertise="expert",
    time_available="urgent"
)

print("Beginner tasks:", len (beginner_tasks))  # More tasks
print("Expert tasks:", len (expert_tasks))      # Fewer tasks
\`\`\`

## Best Practices

1. **Start Coarse, Refine**: Begin with high-level decomposition
2. **Explicit Dependencies**: Always identify what depends on what
3. **Validate Completeness**: Subtasks should cover the whole task
4. **Consider Resources**: Don't plan what you can't execute
5. **Enable Parallelism**: Identify independent tasks
6. **Build in Checkpoints**: Regular validation points
7. **Plan for Failure**: Have fallback options

## Next Steps

You now understand task decomposition and planning. Next, learn:
- Coordinating agent execution
- Managing multi-agent workflows
- Handling inter-agent state
`,
};
