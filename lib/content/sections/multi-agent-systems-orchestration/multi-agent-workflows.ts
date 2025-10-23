/**
 * Multi-Agent Workflows Section
 * Module 7: Multi-Agent Systems & Orchestration
 */

export const multiagentworkflowsSection = {
  id: 'multi-agent-workflows',
  title: 'Multi-Agent Workflows',
  content: `# Multi-Agent Workflows

Master designing and implementing complex workflows where multiple agents collaborate through structured processes.

## Overview: Workflows vs Ad-Hoc Coordination

**Workflows** are predefined, repeatable agent collaboration patterns:

- **Structure**: Clear states and transitions
- **Repeatable**: Same flow for similar tasks
- **Trackable**: Monitor progress through workflow
- **Reliable**: Handle errors at each step
- **Auditable**: Record what happened when

### Workflow Types

**State Machine**: Agent moves through states  
**DAG**: Directed graph of tasks  
**Pipeline**: Sequential stages with validation  
**Loop**: Iterative refinement cycles  
**Conditional**: Branch based on outcomes  

## State Machine Workflows

Agents transition through defined states:

\`\`\`python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Callable, Any, Optional
import asyncio

class WorkflowState(Enum):
    """States in a workflow."""
    INIT = "init"
    RESEARCH = "research"
    PLAN = "plan"
    IMPLEMENT = "implement"
    REVIEW = "review"
    REVISE = "revise"
    COMPLETE = "complete"
    FAILED = "failed"

@dataclass
class StateTransition:
    """A transition between states."""
    from_state: WorkflowState
    to_state: WorkflowState
    condition: Optional[Callable] = None
    action: Optional[Callable] = None

class StateMachineWorkflow:
    """Workflow implemented as state machine."""
    
    def __init__(self, initial_state: WorkflowState = WorkflowState.INIT):
        self.current_state = initial_state
        self.transitions: Dict[WorkflowState, List[StateTransition]] = {}
        self.state_handlers: Dict[WorkflowState, Callable] = {}
        self.state_history: List[WorkflowState] = [initial_state]
        self.context: Dict[str, Any] = {}
    
    def add_transition(
        self,
        from_state: WorkflowState,
        to_state: WorkflowState,
        condition: Optional[Callable] = None,
        action: Optional[Callable] = None
    ):
        """Add a state transition."""
        if from_state not in self.transitions:
            self.transitions[from_state] = []
        
        self.transitions[from_state].append(
            StateTransition(from_state, to_state, condition, action)
        )
    
    def register_handler(self, state: WorkflowState, handler: Callable):
        """Register handler for a state."""
        self.state_handlers[state] = handler
    
    async def execute(self, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow."""
        self.context = initial_context.copy()
        
        while self.current_state not in [WorkflowState.COMPLETE, WorkflowState.FAILED]:
            print(f"Current state: {self.current_state.value}")
            
            # Execute handler for current state
            if self.current_state in self.state_handlers:
                handler = self.state_handlers[self.current_state]
                self.context = await handler(self.context)
            
            # Find next state
            next_state = await self._find_next_state()
            
            if next_state is None:
                print(f"No transition found from {self.current_state}")
                self.current_state = WorkflowState.FAILED
                break
            
            # Transition to next state
            await self._transition_to(next_state)
        
        return {
            "final_state": self.current_state,
            "context": self.context,
            "history": [s.value for s in self.state_history]
        }
    
    async def _find_next_state(self) -> Optional[WorkflowState]:
        """Find next state based on transitions."""
        transitions = self.transitions.get(self.current_state, [])
        
        for transition in transitions:
            # Check condition if exists
            if transition.condition is None or await self._check_condition(transition.condition):
                # Execute action if exists
                if transition.action:
                    await transition.action(self.context)
                
                return transition.to_state
        
        return None
    
    async def _check_condition(self, condition: Callable) -> bool:
        """Check if condition is met."""
        if asyncio.iscoroutinefunction(condition):
            return await condition(self.context)
        else:
            return condition(self.context)
    
    async def _transition_to(self, new_state: WorkflowState):
        """Transition to new state."""
        print(f"Transitioning: {self.current_state.value} -> {new_state.value}")
        self.current_state = new_state
        self.state_history.append(new_state)

# Example: Content creation workflow
async def init_handler(context: Dict) -> Dict:
    """Initialize workflow."""
    print("  [Initializing workflow]")
    context['topic'] = context.get('topic', 'AI')
    return context

async def research_handler(context: Dict) -> Dict:
    """Research phase."""
    print("  [Researcher working...]")
    await asyncio.sleep(1)
    context['research'] = f"Research on {context['topic']}: ..."
    context['research_quality'] = 0.8
    return context

async def plan_handler(context: Dict) -> Dict:
    """Planning phase."""
    print("  [Planner working...]")
    await asyncio.sleep(0.5)
    context['outline'] = ["Intro", "Body", "Conclusion"]
    return context

async def implement_handler(context: Dict) -> Dict:
    """Implementation phase."""
    print("  [Writer working...]")
    await asyncio.sleep(1)
    context['content'] = f"Article based on: {context['research'][:50]}"
    context['content_quality'] = 0.7
    return context

async def review_handler(context: Dict) -> Dict:
    """Review phase."""
    print("  [Reviewer working...]")
    await asyncio.sleep(0.5)
    quality = context.get('content_quality', 0)
    context['review_passed'] = quality >= 0.8
    return context

async def revise_handler(context: Dict) -> Dict:
    """Revise phase."""
    print("  [Revising content...]")
    await asyncio.sleep(0.5)
    # Improve quality
    context['content_quality'] = min(1.0, context['content_quality'] + 0.2)
    return context

# Build workflow
workflow = StateMachineWorkflow(WorkflowState.INIT)

# Register handlers
workflow.register_handler(WorkflowState.INIT, init_handler)
workflow.register_handler(WorkflowState.RESEARCH, research_handler)
workflow.register_handler(WorkflowState.PLAN, plan_handler)
workflow.register_handler(WorkflowState.IMPLEMENT, implement_handler)
workflow.register_handler(WorkflowState.REVIEW, review_handler)
workflow.register_handler(WorkflowState.REVISE, revise_handler)

# Define transitions
workflow.add_transition(WorkflowState.INIT, WorkflowState.RESEARCH)
workflow.add_transition(WorkflowState.RESEARCH, WorkflowState.PLAN)
workflow.add_transition(WorkflowState.PLAN, WorkflowState.IMPLEMENT)
workflow.add_transition(WorkflowState.IMPLEMENT, WorkflowState.REVIEW)

# Conditional transitions from REVIEW
workflow.add_transition(
    WorkflowState.REVIEW,
    WorkflowState.COMPLETE,
    condition=lambda ctx: ctx.get('review_passed', False)
)
workflow.add_transition(
    WorkflowState.REVIEW,
    WorkflowState.REVISE,
    condition=lambda ctx: not ctx.get('review_passed', False)
)

workflow.add_transition(WorkflowState.REVISE, WorkflowState.REVIEW)

# Execute
result = await workflow.execute({"topic": "Quantum Computing"})
print(f"\\nFinal state: {result['final_state'].value}")
print(f"State history: {result['history']}")
\`\`\`

## DAG (Directed Acyclic Graph) Workflows

Tasks as nodes, dependencies as edges:

\`\`\`python
from dataclasses import dataclass, field
from typing import Set

@dataclass
class DAGNode:
    """Node in DAG workflow."""
    id: str
    task: str
    agent: str
    dependencies: Set[str] = field(default_factory=set)
    status: str = "pending"  # pending, running, completed, failed
    result: Any = None

class DAGWorkflow:
    """Workflow as Directed Acyclic Graph."""
    
    def __init__(self):
        self.nodes: Dict[str, DAGNode] = {}
    
    def add_node(self, node: DAGNode):
        """Add node to DAG."""
        if node.id in self.nodes:
            raise ValueError(f"Node {node.id} already exists")
        
        self.nodes[node.id] = node
    
    def add_dependency(self, node_id: str, depends_on: str):
        """Add dependency between nodes."""
        if node_id not in self.nodes or depends_on not in self.nodes:
            raise ValueError("Both nodes must exist")
        
        self.nodes[node_id].dependencies.add(depends_on)
        
        # Check for cycles
        if self._has_cycle():
            self.nodes[node_id].dependencies.remove(depends_on)
            raise ValueError("Adding this dependency creates a cycle")
    
    def _has_cycle(self) -> bool:
        """Check if DAG has cycles."""
        visited = set()
        rec_stack = set()
        
        def visit(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            for dep in self.nodes[node_id].dependencies:
                if dep not in visited:
                    if visit(dep):
                        return True
                elif dep in rec_stack:
                    return True
            
            rec_stack.remove(node_id)
            return False
        
        for node_id in self.nodes:
            if node_id not in visited:
                if visit(node_id):
                    return True
        
        return False
    
    async def execute(self, agents: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DAG workflow."""
        completed = set()
        
        while len(completed) < len(self.nodes):
            # Find nodes ready to execute
            ready = self._get_ready_nodes(completed)
            
            if not ready:
                # Check for failures
                failed = [n for n in self.nodes.values() if n.status == "failed"]
                if failed:
                    raise RuntimeError(f"Workflow failed: {[n.id for n in failed]}")
                else:
                    raise RuntimeError("Workflow stuck - possible circular dependency")
            
            # Execute ready nodes in parallel
            await self._execute_nodes(ready, agents)
            
            # Mark as completed
            for node in ready:
                if node.status == "completed":
                    completed.add(node.id)
        
        return {
            "nodes": {
                node_id: {
                    "task": node.task,
                    "status": node.status,
                    "result": node.result
                }
                for node_id, node in self.nodes.items()
            }
        }
    
    def _get_ready_nodes(self, completed: Set[str]) -> List[DAGNode]:
        """Get nodes ready to execute."""
        ready = []
        
        for node in self.nodes.values():
            if node.status == "pending":
                # Check if all dependencies completed
                if node.dependencies.issubset(completed):
                    ready.append(node)
        
        return ready
    
    async def _execute_nodes(
        self,
        nodes: List[DAGNode],
        agents: Dict[str, Any]
    ):
        """Execute nodes in parallel."""
        tasks = []
        
        for node in nodes:
            node.status = "running"
            tasks.append(self._execute_node(node, agents))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_node(self, node: DAGNode, agents: Dict[str, Any]):
        """Execute single node."""
        agent = agents.get(node.agent)
        
        if not agent:
            node.status = "failed"
            node.result = f"Agent {node.agent} not found"
            return
        
        try:
            print(f"[{node.agent}] Executing: {node.task}")
            result = await agent.execute(node.task)
            node.status = "completed"
            node.result = result
        except Exception as e:
            node.status = "failed"
            node.result = str(e)
    
    def visualize(self) -> str:
        """Visualize DAG."""
        lines = ["DAG Workflow:", ""]
        
        # Sort by dependency depth
        sorted_nodes = self._topological_sort()
        
        for node_id in sorted_nodes:
            node = self.nodes[node_id]
            status_icon = {
                "pending": "⭕",
                "running": "🔄",
                "completed": "✅",
                "failed": "❌"
            }.get(node.status, "?")
            
            deps = ", ".join(node.dependencies) if node.dependencies else "None"
            lines.append(f"{status_icon} {node.id}: {node.task}")
            lines.append(f"   Agent: {node.agent}, Dependencies: {deps}")
            lines.append("")
        
        return "\\n".join(lines)
    
    def _topological_sort(self) -> List[str]:
        """Topological sort of nodes."""
        in_degree = {node_id: 0 for node_id in self.nodes}
        
        for node in self.nodes.values():
            for dep in node.dependencies:
                in_degree[dep] += 1
        
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            node_id = queue.pop(0)
            result.append(node_id)
            
            # Reduce in-degree for dependent nodes
            for other_id, other_node in self.nodes.items():
                if node_id in other_node.dependencies:
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        queue.append(other_id)
        
        return result

# Example: Software development workflow
dag = DAGWorkflow()

# Add nodes
dag.add_node(DAGNode("requirements", "Gather requirements", "researcher"))
dag.add_node(DAGNode("design", "Design architecture", "architect"))
dag.add_node(DAGNode("backend", "Implement backend", "backend_dev"))
dag.add_node(DAGNode("frontend", "Implement frontend", "frontend_dev"))
dag.add_node(DAGNode("integration", "Integrate components", "integrator"))
dag.add_node(DAGNode("testing", "Run tests", "tester"))

# Add dependencies
dag.add_dependency("design", "requirements")
dag.add_dependency("backend", "design")
dag.add_dependency("frontend", "design")
dag.add_dependency("integration", "backend")
dag.add_dependency("integration", "frontend")
dag.add_dependency("testing", "integration")

print(dag.visualize())

# Execute
agents = {
    "researcher": Agent("Researcher", lambda t: f"Requirements: {t}"),
    "architect": Agent("Architect", lambda t: f"Design: {t}"),
    "backend_dev": Agent("Backend", lambda t: f"Backend: {t}"),
    "frontend_dev": Agent("Frontend", lambda t: f"Frontend: {t}"),
    "integrator": Agent("Integrator", lambda t: f"Integrated: {t}"),
    "tester": Agent("Tester", lambda t: f"Tests: {t}")
}

result = await dag.execute(agents)
\`\`\`

## Pipeline Workflows

Sequential stages with validation:

\`\`\`python
@dataclass
class PipelineStage:
    """Stage in a pipeline."""
    name: str
    agent: Any
    validator: Optional[Callable] = None
    retry_on_fail: int = 0

class PipelineWorkflow:
    """Pipeline workflow with validation."""
    
    def __init__(self, stages: List[PipelineStage]):
        self.stages = stages
        self.stage_results: List[Dict] = []
    
    async def execute(self, initial_input: Any) -> Dict[str, Any]:
        """Execute pipeline."""
        current_input = initial_input
        
        for i, stage in enumerate(self.stages):
            print(f"\\nStage {i+1}/{len(self.stages)}: {stage.name}")
            
            # Execute stage with retries
            result = await self._execute_stage_with_retry(
                stage,
                current_input
            )
            
            # Store result
            self.stage_results.append(result)
            
            # Check if stage succeeded
            if not result['success']:
                return {
                    "success": False,
                    "failed_stage": stage.name,
                    "stage_results": self.stage_results
                }
            
            # Output becomes next input
            current_input = result['output']
        
        return {
            "success": True,
            "final_output": current_input,
            "stage_results": self.stage_results
        }
    
    async def _execute_stage_with_retry(
        self,
        stage: PipelineStage,
        input: Any
    ) -> Dict[str, Any]:
        """Execute stage with retry logic."""
        attempts = 0
        max_attempts = stage.retry_on_fail + 1
        
        while attempts < max_attempts:
            try:
                # Execute agent
                output = await stage.agent.execute(input)
                
                # Validate if validator exists
                if stage.validator:
                    is_valid = await self._validate(stage.validator, output)
                    if not is_valid:
                        if attempts < max_attempts - 1:
                            print(f"  Validation failed, retrying...")
                            attempts += 1
                            continue
                        else:
                            return {
                                "stage": stage.name,
                                "success": False,
                                "error": "Validation failed",
                                "output": None
                            }
                
                # Success
                return {
                    "stage": stage.name,
                    "success": True,
                    "output": output,
                    "attempts": attempts + 1
                }
            
            except Exception as e:
                if attempts < max_attempts - 1:
                    print(f"  Error: {e}, retrying...")
                    attempts += 1
                else:
                    return {
                        "stage": stage.name,
                        "success": False,
                        "error": str(e),
                        "output": None
                    }
    
    async def _validate(self, validator: Callable, output: Any) -> bool:
        """Run validator."""
        if asyncio.iscoroutinefunction(validator):
            return await validator(output)
        else:
            return validator(output)

# Example: Content pipeline with validation
def validate_research(research: str) -> bool:
    """Validate research quality."""
    return len(research) > 100  # Simple check

def validate_article(article: str) -> bool:
    """Validate article quality."""
    return len(article) > 200 and "." in article

pipeline = PipelineWorkflow([
    PipelineStage(
        "Research",
        agent=Agent("Researcher", lambda t: f"Research on {t}: ..."),
        validator=validate_research,
        retry_on_fail=2
    ),
    PipelineStage(
        "Write",
        agent=Agent("Writer", lambda r: f"Article based on: {r}..."),
        validator=validate_article,
        retry_on_fail=1
    ),
    PipelineStage(
        "Review",
        agent=Agent("Reviewer", lambda a: f"Reviewed: {a}"),
        validator=None  # No validation
    )
])

result = await pipeline.execute("quantum computing")
print(f"\\nPipeline success: {result['success']}")
if result['success']:
    print(f"Final output: {result['final_output'][:100]}...")
\`\`\`

## Loop Workflows

Iterative refinement:

\`\`\`python
class LoopWorkflow:
    """Workflow with iterative refinement."""
    
    def __init__(
        self,
        generator: Any,
        evaluator: Any,
        refiner: Any,
        max_iterations: int = 5,
        quality_threshold: float = 0.8
    ):
        self.generator = generator
        self.evaluator = evaluator
        self.refiner = refiner
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
    
    async def execute(self, task: str) -> Dict[str, Any]:
        """Execute iterative workflow."""
        iterations = []
        
        # Initial generation
        current_output = await self.generator.execute(task)
        
        for i in range(self.max_iterations):
            print(f"\\nIteration {i+1}/{self.max_iterations}")
            
            # Evaluate
            evaluation = await self.evaluator.execute(current_output)
            quality = evaluation.get('score', 0)
            
            iterations.append({
                "iteration": i + 1,
                "output": current_output,
                "quality": quality,
                "feedback": evaluation.get('feedback', '')
            })
            
            print(f"  Quality score: {quality:.2f}")
            
            # Check if good enough
            if quality >= self.quality_threshold:
                print("  Quality threshold reached!")
                return {
                    "success": True,
                    "final_output": current_output,
                    "iterations": iterations,
                    "converged_at": i + 1
                }
            
            # Refine for next iteration
            print("  Refining...")
            current_output = await self.refiner.execute({
                "output": current_output,
                "feedback": evaluation.get('feedback', '')
            })
        
        # Max iterations reached
        return {
            "success": False,
            "final_output": current_output,
            "iterations": iterations,
            "message": "Max iterations reached without convergence"
        }

# Example: Iterative article refinement
class Generator:
    name = "Generator"
    async def execute(self, task: str) -> str:
        return f"Draft article on {task}: Basic content..."

class Evaluator:
    name = "Evaluator"
    async def execute(self, output: str) -> Dict:
        # Simulate evaluation
        import random
        score = min(1.0, len(output) / 500 + random.random() * 0.2)
        return {
            "score": score,
            "feedback": "Add more detail" if score < 0.8 else "Good"
        }

class Refiner:
    name = "Refiner"
    async def execute(self, data: Dict) -> str:
        output = data['output']
        feedback = data['feedback']
        # Simulate refinement
        return output + f" [Refined based on: {feedback}]..."

loop_workflow = LoopWorkflow(
    generator=Generator(),
    evaluator=Evaluator(),
    refiner=Refiner(),
    max_iterations=5,
    quality_threshold=0.8
)

result = await loop_workflow.execute("quantum computing")
print(f"\\nSuccess: {result['success']}")
print(f"Iterations: {len(result['iterations'])}")
print(f"Final quality: {result['iterations'][-1]['quality']:.2f}")
\`\`\`

## Conditional Workflows

Branch based on outcomes:

\`\`\`python
class ConditionalWorkflow:
    """Workflow with conditional branching."""
    
    def __init__(self):
        self.branches: Dict[str, Callable] = {}
    
    def add_branch(self, name: str, handler: Callable):
        """Add a branch handler."""
        self.branches[name] = handler
    
    async def execute(
        self,
        input: Any,
        router: Callable
    ) -> Dict[str, Any]:
        """Execute workflow with routing."""
        # Determine which branch to take
        branch_name = await router(input)
        
        print(f"Routing to branch: {branch_name}")
        
        if branch_name not in self.branches:
            raise ValueError(f"Branch {branch_name} not found")
        
        # Execute selected branch
        handler = self.branches[branch_name]
        result = await handler(input)
        
        return {
            "branch_taken": branch_name,
            "result": result
        }

# Example: Content type routing
async def determine_content_type(request: Dict) -> str:
    """Route based on content type."""
    content_type = request.get('type', 'article')
    
    if content_type == 'code':
        return 'code_branch'
    elif content_type == 'article':
        return 'article_branch'
    elif content_type == 'video':
        return 'video_branch'
    else:
        return 'default_branch'

async def code_branch_handler(request: Dict) -> str:
    print("  [Code workflow: Research -> Code -> Test]")
    return "Code generated"

async def article_branch_handler(request: Dict) -> str:
    print("  [Article workflow: Research -> Write -> Review]")
    return "Article written"

async def video_branch_handler(request: Dict) -> str:
    print("  [Video workflow: Script -> Record -> Edit]")
    return "Video created"

conditional = ConditionalWorkflow()
conditional.add_branch('code_branch', code_branch_handler)
conditional.add_branch('article_branch', article_branch_handler)
conditional.add_branch('video_branch', video_branch_handler)

# Execute with different inputs
result1 = await conditional.execute(
    {'type': 'code', 'topic': 'sorting'},
    router=determine_content_type
)

result2 = await conditional.execute(
    {'type': 'article', 'topic': 'AI'},
    router=determine_content_type
)
\`\`\`

## Best Practices

1. **Clear States**: Define explicit workflow states
2. **Idempotent Steps**: Steps should be repeatable safely
3. **Checkpointing**: Save state at key points
4. **Error Recovery**: Handle failures at each stage
5. **Monitoring**: Track progress through workflow
6. **Validation**: Verify outputs between stages
7. **Versioning**: Version your workflows

## Next Steps

You now understand workflows. Next, learn:
- Inter-agent memory and state
- LangGraph for workflows
- Production debugging
`,
};
