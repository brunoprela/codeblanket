/**
 * Agent Coordination Strategies Section
 * Module 7: Multi-Agent Systems & Orchestration
 */

export const agentcoordinationstrategiesSection = {
  id: 'agent-coordination-strategies',
  title: 'Agent Coordination Strategies',
  content: `# Agent Coordination Strategies

Master different approaches to coordinating multiple agents for efficient collaboration.

## Overview: Coordination Patterns

Agents need coordination to work together effectively:

- **Sequential**: One after another (like assembly line)
- **Parallel**: Multiple agents work simultaneously
- **Hierarchical**: Manager directs workers
- **Consensus**: Agents agree on decisions
- **Competitive**: Agents compete for best solution

### Choosing a Strategy

**Sequential**: When tasks have strict dependencies  
**Parallel**: When tasks are independent  
**Hierarchical**: When need central control  
**Consensus**: When need agreement  
**Competitive**: When want best of multiple attempts  

## Sequential Coordination

Agents execute in order, each building on previous:

\`\`\`python
from typing import List, Any, Optional
from dataclasses import dataclass
import asyncio

@dataclass
class SequentialResult:
    """Result from sequential execution."""
    final_output: Any
    step_outputs: List[Any]
    total_time: float
    step_times: List[float]

class SequentialCoordinator:
    """Coordinates agents in sequence."""
    
    def __init__(self, agents: List[Any]):
        self.agents = agents
    
    async def execute (self, initial_input: Any) -> SequentialResult:
        """Execute agents sequentially."""
        import time
        
        start_time = time.time()
        step_outputs = []
        step_times = []
        
        current_input = initial_input
        
        for i, agent in enumerate (self.agents):
            step_start = time.time()
            
            print(f"[Step {i+1}/{len (self.agents)}] {agent.name}...")
            
            # Execute agent
            output = await agent.execute (current_input)
            
            step_time = time.time() - step_start
            
            step_outputs.append (output)
            step_times.append (step_time)
            
            # Output becomes next input
            current_input = output
            
            print(f"  Completed in {step_time:.2f}s")
        
        total_time = time.time() - start_time
        
        return SequentialResult(
            final_output=current_input,
            step_outputs=step_outputs,
            total_time=total_time,
            step_times=step_times
        )
    
    async def execute_with_validation(
        self,
        initial_input: Any,
        validators: Optional[List[callable]] = None
    ) -> SequentialResult:
        """Execute with validation after each step."""
        import time
        
        start_time = time.time()
        step_outputs = []
        step_times = []
        
        current_input = initial_input
        
        for i, agent in enumerate (self.agents):
            step_start = time.time()
            
            # Execute agent
            output = await agent.execute (current_input)
            
            # Validate if validator provided
            if validators and i < len (validators) and validators[i]:
                is_valid = validators[i](output)
                if not is_valid:
                    raise ValueError (f"Validation failed after {agent.name}")
            
            step_time = time.time() - step_start
            
            step_outputs.append (output)
            step_times.append (step_time)
            current_input = output
        
        total_time = time.time() - start_time
        
        return SequentialResult(
            final_output=current_input,
            step_outputs=step_outputs,
            total_time=total_time,
            step_times=step_times
        )

# Example: Research → Write → Review → Edit
class Agent:
    def __init__(self, name: str, process_fn: callable):
        self.name = name
        self.process_fn = process_fn
    
    async def execute (self, input: Any) -> Any:
        return await self.process_fn (input)

async def research (topic: str) -> str:
    await asyncio.sleep(1)  # Simulate work
    return f"Research findings on {topic}: ..."

async def write (research: str) -> str:
    await asyncio.sleep(1.5)
    return f"Article based on: {research[:50]}..."

async def review (article: str) -> str:
    await asyncio.sleep(1)
    return f"Reviewed article: {article[:50]}..."

async def edit (reviewed: str) -> str:
    await asyncio.sleep(0.5)
    return f"Final edited: {reviewed[:50]}..."

# Usage
coordinator = SequentialCoordinator([
    Agent("Researcher", research),
    Agent("Writer", write),
    Agent("Reviewer", review),
    Agent("Editor", edit)
])

result = await coordinator.execute("quantum computing")
print(f"Final output: {result.final_output}")
print(f"Total time: {result.total_time:.2f}s")
\`\`\`

## Parallel Coordination

Multiple agents work simultaneously:

\`\`\`python
from typing import Dict

@dataclass
class ParallelResult:
    """Result from parallel execution."""
    outputs: Dict[str, Any]
    total_time: float
    individual_times: Dict[str, float]
    failures: Dict[str, str]

class ParallelCoordinator:
    """Coordinates agents in parallel."""
    
    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents  # {"name": agent}
    
    async def execute_all (self, input: Any) -> ParallelResult:
        """Execute all agents in parallel."""
        import time
        
        start_time = time.time()
        
        # Create tasks for all agents
        tasks = {
            name: self._execute_with_timing (name, agent, input)
            for name, agent in self.agents.items()
        }
        
        # Wait for all to complete
        results = await asyncio.gather(
            *tasks.values(),
            return_exceptions=True
        )
        
        # Process results
        outputs = {}
        individual_times = {}
        failures = {}
        
        for name, result in zip (tasks.keys(), results):
            if isinstance (result, Exception):
                failures[name] = str (result)
            else:
                outputs[name] = result['output']
                individual_times[name] = result['time']
        
        total_time = time.time() - start_time
        
        return ParallelResult(
            outputs=outputs,
            total_time=total_time,
            individual_times=individual_times,
            failures=failures
        )
    
    async def _execute_with_timing(
        self,
        name: str,
        agent: Any,
        input: Any
    ) -> Dict[str, Any]:
        """Execute agent and track time."""
        import time
        
        start = time.time()
        output = await agent.execute (input)
        elapsed = time.time() - start
        
        return {'output': output, 'time': elapsed}
    
    async def execute_with_timeout(
        self,
        input: Any,
        timeout: float = 30.0
    ) -> ParallelResult:
        """Execute with timeout for each agent."""
        import time
        
        start_time = time.time()
        
        outputs = {}
        individual_times = {}
        failures = {}
        
        for name, agent in self.agents.items():
            try:
                result = await asyncio.wait_for(
                    self._execute_with_timing (name, agent, input),
                    timeout=timeout
                )
                outputs[name] = result['output']
                individual_times[name] = result['time']
            except asyncio.TimeoutError:
                failures[name] = f"Timeout after {timeout}s"
            except Exception as e:
                failures[name] = str (e)
        
        total_time = time.time() - start_time
        
        return ParallelResult(
            outputs=outputs,
            total_time=total_time,
            individual_times=individual_times,
            failures=failures
        )

# Example: Analyze text from multiple perspectives
async def analyze_sentiment (text: str) -> Dict:
    await asyncio.sleep(1)
    return {"sentiment": "positive", "score": 0.8}

async def extract_keywords (text: str) -> Dict:
    await asyncio.sleep(1.5)
    return {"keywords": ["AI", "technology"]}

async def detect_language (text: str) -> Dict:
    await asyncio.sleep(0.5)
    return {"language": "en", "confidence": 0.99}

# Usage
parallel_coord = ParallelCoordinator({
    "sentiment": Agent("Sentiment", analyze_sentiment),
    "keywords": Agent("Keywords", extract_keywords),
    "language": Agent("Language", detect_language)
})

result = await parallel_coord.execute_all("I love AI technology!")

print("Outputs:", result.outputs)
print(f"Total time: {result.total_time:.2f}s")  # ~1.5s (longest)
print(f"Individual times: {result.individual_times}")
# vs Sequential: would be 1 + 1.5 + 0.5 = 3s
\`\`\`

## Hierarchical Coordination

Manager agent coordinates worker agents:

\`\`\`python
class HierarchicalCoordinator:
    """Manager-worker coordination."""
    
    def __init__(self, manager: Any, workers: Dict[str, Any]):
        self.manager = manager
        self.workers = workers
    
    async def execute (self, goal: str) -> Dict[str, Any]:
        """Execute with manager coordinating workers."""
        # 1. Manager creates plan
        print("[Manager] Planning...")
        plan = await self.manager.create_plan (goal, list (self.workers.keys()))
        
        # 2. Execute plan stages
        results = []
        for stage in plan['stages']:
            print(f"\\n[Manager] Executing Stage {stage['id']}...")
            stage_results = await self._execute_stage (stage)
            results.extend (stage_results)
        
        # 3. Manager synthesizes results
        print("\\n[Manager] Synthesizing results...")
        final_result = await self.manager.synthesize (results)
        
        return {
            "plan": plan,
            "results": results,
            "final_result": final_result
        }
    
    async def _execute_stage (self, stage: Dict) -> List[Dict]:
        """Execute all tasks in a stage."""
        tasks = []
        
        for task in stage['tasks']:
            worker = self.workers.get (task['worker'])
            if worker:
                tasks.append (self._execute_task (worker, task))
        
        # Execute stage tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [
            r if not isinstance (r, Exception) else {"error": str (r)}
            for r in results
        ]
    
    async def _execute_task (self, worker: Any, task: Dict) -> Dict:
        """Execute single task."""
        print(f"  [{worker.name}] {task['description']}")
        output = await worker.execute (task['description'])
        return {
            "task": task['description'],
            "worker": worker.name,
            "output": output
        }

# Example Manager
class ManagerAgent:
    name = "Manager"
    
    async def create_plan (self, goal: str, workers: List[str]) -> Dict:
        """Create execution plan."""
        # Simplified - in reality, use LLM
        return {
            "goal": goal,
            "stages": [
                {
                    "id": 1,
                    "tasks": [
                        {"description": "Research topic", "worker": "researcher"}
                    ]
                },
                {
                    "id": 2,
                    "tasks": [
                        {"description": "Write code", "worker": "coder"},
                        {"description": "Write tests", "worker": "tester"}
                    ]
                },
                {
                    "id": 3,
                    "tasks": [
                        {"description": "Review everything", "worker": "reviewer"}
                    ]
                }
            ]
        }
    
    async def synthesize (self, results: List[Dict]) -> str:
        """Synthesize results."""
        return "Project completed successfully with all tasks done."

# Usage
manager = ManagerAgent()
workers = {
    "researcher": Agent("Researcher", lambda x: f"Research: {x}"),
    "coder": Agent("Coder", lambda x: f"Code: {x}"),
    "tester": Agent("Tester", lambda x: f"Tests: {x}"),
    "reviewer": Agent("Reviewer", lambda x: f"Review: {x}")
}

hierarchical = HierarchicalCoordinator (manager, workers)
result = await hierarchical.execute("Build authentication system")
\`\`\`

## Consensus Coordination

Agents reach agreement:

\`\`\`python
from typing import List, Callable

class ConsensusCoordinator:
    """Coordinates agents to reach consensus."""
    
    def __init__(self, agents: List[Any]):
        self.agents = agents
    
    async def reach_consensus(
        self,
        question: str,
        consensus_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Get consensus from agents."""
        # 1. Get opinions from all agents
        opinions = await self._gather_opinions (question)
        
        # 2. Find consensus
        consensus = self._find_consensus (opinions, consensus_threshold)
        
        # 3. If no consensus, iterate
        if not consensus['reached']:
            consensus = await self._iterate_to_consensus(
                question,
                opinions,
                consensus_threshold
            )
        
        return consensus
    
    async def _gather_opinions (self, question: str) -> List[Dict]:
        """Get opinion from each agent."""
        tasks = [
            agent.execute (question)
            for agent in self.agents
        ]
        
        results = await asyncio.gather(*tasks)
        
        opinions = []
        for agent, result in zip (self.agents, results):
            opinions.append({
                "agent": agent.name,
                "opinion": result,
                "confidence": getattr (result, 'confidence', 1.0)
            })
        
        return opinions
    
    def _find_consensus(
        self,
        opinions: List[Dict],
        threshold: float
    ) -> Dict[str, Any]:
        """Find if consensus exists."""
        # Simple majority voting
        # Count similar opinions
        opinion_groups = {}
        
        for op in opinions:
            opinion_text = str (op['opinion'])
            if opinion_text not in opinion_groups:
                opinion_groups[opinion_text] = []
            opinion_groups[opinion_text].append (op)
        
        # Find majority
        total = len (opinions)
        for opinion_text, group in opinion_groups.items():
            agreement_ratio = len (group) / total
            
            if agreement_ratio >= threshold:
                return {
                    "reached": True,
                    "consensus_opinion": opinion_text,
                    "agreement_ratio": agreement_ratio,
                    "supporting_agents": [op['agent'] for op in group],
                    "dissenting_agents": [
                        op['agent'] for op in opinions
                        if op not in group
                    ]
                }
        
        return {
            "reached": False,
            "opinion_groups": {
                opinion: [op['agent'] for op in group]
                for opinion, group in opinion_groups.items()
            }
        }
    
    async def _iterate_to_consensus(
        self,
        question: str,
        initial_opinions: List[Dict],
        threshold: float,
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """Iterate to reach consensus."""
        current_opinions = initial_opinions
        
        for iteration in range (max_iterations):
            print(f"Consensus iteration {iteration + 1}...")
            
            # Share opinions with all agents
            opinion_summary = self._summarize_opinions (current_opinions)
            
            # Get revised opinions
            revised = await self._gather_revised_opinions(
                question,
                opinion_summary
            )
            
            # Check for consensus
            consensus = self._find_consensus (revised, threshold)
            
            if consensus['reached']:
                consensus['iterations'] = iteration + 1
                return consensus
            
            current_opinions = revised
        
        # Failed to reach consensus
        return {
            "reached": False,
            "iterations": max_iterations,
            "final_opinions": current_opinions
        }
    
    def _summarize_opinions (self, opinions: List[Dict]) -> str:
        """Summarize all opinions."""
        summary = "Current opinions:\\n"
        for op in opinions:
            summary += f"- {op['agent']}: {op['opinion']}\\n"
        return summary
    
    async def _gather_revised_opinions(
        self,
        question: str,
        opinion_summary: str
    ) -> List[Dict]:
        """Get revised opinions after seeing others'."""
        revised_question = f"""{question}

Other agents' opinions:
{opinion_summary}

Consider these perspectives and provide your revised opinion."""
        
        return await self._gather_opinions (revised_question)

# Usage
class OpinionAgent:
    def __init__(self, name: str, bias: str):
        self.name = name
        self.bias = bias
    
    async def execute (self, question: str) -> str:
        # Simulate agent forming opinion
        await asyncio.sleep(0.5)
        return f"Opinion from {self.bias} perspective: ..."

consensus_coord = ConsensusCoordinator([
    OpinionAgent("Agent1", "optimistic"),
    OpinionAgent("Agent2", "realistic"),
    OpinionAgent("Agent3", "pessimistic"),
    OpinionAgent("Agent4", "realistic")
])

consensus = await consensus_coord.reach_consensus(
    "Should we proceed with the project?",
    consensus_threshold=0.5
)

print("Consensus reached:", consensus['reached'])
if consensus['reached']:
    print("Agreement:", consensus['consensus_opinion'])
\`\`\`

## Competitive Coordination

Multiple agents compete, best solution wins:

\`\`\`python
class CompetitiveCoordinator:
    """Coordinates competing agents."""
    
    def __init__(self, agents: List[Any], judge: Callable):
        self.agents = agents
        self.judge = judge  # Function to evaluate solutions
    
    async def compete (self, task: str) -> Dict[str, Any]:
        """Have agents compete on task."""
        # 1. All agents attempt task in parallel
        solutions = await self._gather_solutions (task)
        
        # 2. Judge evaluates all solutions
        scored_solutions = await self._score_solutions (solutions)
        
        # 3. Select winner
        winner = max (scored_solutions, key=lambda x: x['score'])
        
        return {
            "task": task,
            "all_solutions": scored_solutions,
            "winner": winner,
            "diversity_score": self._calculate_diversity (scored_solutions)
        }
    
    async def _gather_solutions (self, task: str) -> List[Dict]:
        """Get solution from each agent."""
        tasks = [
            self._get_solution (agent, task)
            for agent in self.agents
        ]
        
        solutions = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [
            s for s in solutions
            if not isinstance (s, Exception)
        ]
    
    async def _get_solution (self, agent: Any, task: str) -> Dict:
        """Get solution from single agent."""
        import time
        
        start = time.time()
        solution = await agent.execute (task)
        elapsed = time.time() - start
        
        return {
            "agent": agent.name,
            "solution": solution,
            "time": elapsed
        }
    
    async def _score_solutions (self, solutions: List[Dict]) -> List[Dict]:
        """Score all solutions."""
        for solution in solutions:
            # Use judge function to score
            score = await self.judge (solution['solution'])
            solution['score'] = score
        
        return solutions
    
    def _calculate_diversity (self, solutions: List[Dict]) -> float:
        """Calculate diversity of solutions."""
        # Simple: ratio of unique solutions
        unique = len (set (str (s['solution']) for s in solutions))
        total = len (solutions)
        return unique / total if total > 0 else 0

# Example: Multiple agents solve coding problem
async def judge_code_solution (code: str) -> float:
    """Judge quality of code solution."""
    # In reality, run tests and analyze
    score = 0.0
    
    # Check length (not too short, not too long)
    if 50 < len (code) < 500:
        score += 0.3
    
    # Check for error handling
    if "try" in code or "except" in code:
        score += 0.3
    
    # Check for documentation
    if '"""' in code or "''" in code:
        score += 0.2
    
    # Check for type hints
    if ":" in code and "->" in code:
        score += 0.2
    
    return score

competitive = CompetitiveCoordinator(
    agents=[
        Agent("Coder1", lambda t: "def solve(): ..."),
        Agent("Coder2", lambda t: 'def solve(): """Docstring""" ...'),
        Agent("Coder3", lambda t: "def solve() -> int: try: ... except: ...")
    ],
    judge=judge_code_solution
)

result = await competitive.compete("Write a function to calculate fibonacci")
print(f"Winner: {result['winner']['agent']}")
print(f"Score: {result['winner']['score']}")
\`\`\`

## Adaptive Coordination

Switch strategies based on context:

\`\`\`python
class AdaptiveCoordinator:
    """Adapts coordination strategy based on context."""
    
    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
        self.sequential = SequentialCoordinator (list (agents.values()))
        self.parallel = ParallelCoordinator (agents)
    
    async def execute (self, task: str, context: Dict[str, Any]) -> Dict:
        """Choose and execute appropriate strategy."""
        strategy = self._choose_strategy (task, context)
        
        print(f"Using {strategy} coordination...")
        
        if strategy == "sequential":
            return await self.sequential.execute (task)
        elif strategy == "parallel":
            return await self.parallel.execute_all (task)
        elif strategy == "hierarchical":
            # Create manager on the fly
            manager = ManagerAgent()
            hierarchical = HierarchicalCoordinator (manager, self.agents)
            return await hierarchical.execute (task)
        else:
            raise ValueError (f"Unknown strategy: {strategy}")
    
    def _choose_strategy (self, task: str, context: Dict[str, Any]) -> str:
        """Choose coordination strategy based on context."""
        # Rule-based decision
        
        if context.get("has_strict_dependencies"):
            return "sequential"
        
        if context.get("time_critical") and context.get("independent_subtasks"):
            return "parallel"
        
        if context.get("complex") and len (self.agents) > 3:
            return "hierarchical"
        
        # Default
        return "sequential"

# Usage
adaptive = AdaptiveCoordinator({
    "agent1": Agent("Agent1", lambda x: x),
    "agent2": Agent("Agent2", lambda x: x),
    "agent3": Agent("Agent3", lambda x: x)
})

# Context determines strategy
result1 = await adaptive.execute("Task A", {
    "has_strict_dependencies": True
})  # Uses sequential

result2 = await adaptive.execute("Task B", {
    "time_critical": True,
    "independent_subtasks": True
})  # Uses parallel
\`\`\`

## Best Practices

1. **Match Strategy to Task**: Choose appropriate coordination
2. **Handle Failures**: One agent failing shouldn't break everything
3. **Track Progress**: Monitor what each agent is doing
4. **Enable Cancellation**: Stop execution if needed
5. **Optimize Resources**: Don't waste agent time
6. **Log Coordination**: Track decisions and handoffs
7. **Test Strategies**: Benchmark different approaches

## Next Steps

You now understand coordination strategies. Next, learn:
- Building multi-agent workflows
- Managing inter-agent state
- Using frameworks like LangGraph
`,
};
