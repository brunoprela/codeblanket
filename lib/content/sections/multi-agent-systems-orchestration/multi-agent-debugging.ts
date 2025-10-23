/**
 * Multi-Agent Debugging Section
 * Module 7: Multi-Agent Systems & Orchestration
 */

export const multiagentdebuggingSection = {
  id: 'multi-agent-debugging',
  title: 'Multi-Agent Debugging',
  content: `# Multi-Agent Debugging

Master debugging complex multi-agent systems, identifying bottlenecks, and fixing failures.

## Overview: Why Debugging is Harder

Multi-agent systems are complex:

- **Multiple Actors**: Hard to track who did what
- **Async Execution**: Events happen in parallel
- **Emergent Behavior**: Unexpected interactions
- **Distributed State**: State spread across agents
- **Cascading Failures**: One failure affects many

### Debugging Strategies

**Logging**: Track all agent actions  
**Tracing**: Follow execution paths  
**Visualization**: See agent interactions  
**Replay**: Reproduce failures  
**Breakpoints**: Pause execution  

## Comprehensive Logging

\`\`\`python
import logging
from typing import Dict, Any, List
from dataclasses import dataclass, field
import time
import json

@dataclass
class AgentLog:
    """Log entry for agent action."""
    timestamp: float
    agent_name: str
    action: str
    input: Any
    output: Any
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

class AgentLogger:
    """Centralized logger for multi-agent systems."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.logs: List[AgentLog] = []
        self.log_file = log_file
        
        # Set up Python logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(name)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file) if log_file else logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("MultiAgent")
    
    def log_action(
        self,
        agent_name: str,
        action: str,
        input: Any,
        output: Any,
        duration: float,
        metadata: Optional[Dict] = None,
        error: Optional[str] = None
    ):
        """Log agent action."""
        log_entry = AgentLog(
            timestamp=time.time(),
            agent_name=agent_name,
            action=action,
            input=input,
            output=output,
            duration=duration,
            metadata=metadata or {},
            error=error
        )
        
        self.logs.append(log_entry)
        
        # Also log to Python logger
        if error:
            self.logger.error(
                f"{agent_name}.{action} FAILED: {error} ({duration:.2f}s)"
            )
        else:
            self.logger.info(
                f"{agent_name}.{action} ({duration:.2f}s)"
            )
    
    def get_logs(
        self,
        agent_name: Optional[str] = None,
        action: Optional[str] = None,
        has_error: Optional[bool] = None
    ) -> List[AgentLog]:
        """Filter and retrieve logs."""
        filtered = self.logs
        
        if agent_name:
            filtered = [l for l in filtered if l.agent_name == agent_name]
        
        if action:
            filtered = [l for l in filtered if l.action == action]
        
        if has_error is not None:
            filtered = [
                l for l in filtered
                if (l.error is not None) == has_error
            ]
        
        return filtered
    
    def get_timeline(self, agent_name: Optional[str] = None) -> str:
        """Get timeline of events."""
        logs = self.get_logs(agent_name=agent_name)
        
        timeline = []
        for log in logs:
            status = "❌" if log.error else "✅"
            timeline.append(
                f"{status} {log.timestamp:.2f}s: {log.agent_name}.{log.action} "
                f"({log.duration:.2f}s)"
            )
        
        return "\\n".join(timeline)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self.logs:
            return {}
        
        total_duration = sum(l.duration for l in self.logs)
        errors = [l for l in self.logs if l.error]
        
        agent_stats = {}
        for log in self.logs:
            if log.agent_name not in agent_stats:
                agent_stats[log.agent_name] = {
                    "actions": 0,
                    "total_time": 0,
                    "errors": 0
                }
            
            agent_stats[log.agent_name]["actions"] += 1
            agent_stats[log.agent_name]["total_time"] += log.duration
            if log.error:
                agent_stats[log.agent_name]["errors"] += 1
        
        return {
            "total_actions": len(self.logs),
            "total_duration": total_duration,
            "total_errors": len(errors),
            "agent_statistics": agent_stats
        }
    
    def export_json(self, filepath: str):
        """Export logs to JSON."""
        logs_data = [
            {
                "timestamp": l.timestamp,
                "agent_name": l.agent_name,
                "action": l.action,
                "input": str(l.input)[:100],  # Truncate
                "output": str(l.output)[:100],
                "duration": l.duration,
                "metadata": l.metadata,
                "error": l.error
            }
            for l in self.logs
        ]
        
        with open(filepath, 'w') as f:
            json.dump(logs_data, f, indent=2)

# Usage
logger = AgentLogger("multi_agent.log")

# In each agent, log actions
class LoggedAgent:
    def __init__(self, name: str, logger: AgentLogger):
        self.name = name
        self.logger = logger
    
    async def execute(self, input: Any) -> Any:
        """Execute with logging."""
        start_time = time.time()
        error = None
        output = None
        
        try:
            output = await self._do_work(input)
        except Exception as e:
            error = str(e)
            raise
        finally:
            duration = time.time() - start_time
            self.logger.log_action(
                agent_name=self.name,
                action="execute",
                input=input,
                output=output,
                duration=duration,
                error=error
            )
        
        return output
    
    async def _do_work(self, input: Any) -> Any:
        """Actual work."""
        import asyncio
        await asyncio.sleep(0.5)
        return f"Processed: {input}"

# Use
agent1 = LoggedAgent("Agent1", logger)
agent2 = LoggedAgent("Agent2", logger)

await agent1.execute("task1")
await agent2.execute("task2")

# View logs
print(logger.get_timeline())
print(logger.get_statistics())
\`\`\`

## Distributed Tracing

\`\`\`python
import uuid
from typing import Optional

@dataclass
class TraceContext:
    """Context for distributed tracing."""
    trace_id: str
    parent_span_id: Optional[str]
    span_id: str
    
    @classmethod
    def new_trace(cls) -> 'TraceContext':
        """Start new trace."""
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        return cls(trace_id=trace_id, parent_span_id=None, span_id=span_id)
    
    def child_span(self) -> 'TraceContext':
        """Create child span."""
        return TraceContext(
            trace_id=self.trace_id,
            parent_span_id=self.span_id,
            span_id=str(uuid.uuid4())
        )

class TracedAgent:
    """Agent with distributed tracing."""
    
    def __init__(self, name: str):
        self.name = name
    
    async def execute(
        self,
        input: Any,
        trace_context: Optional[TraceContext] = None
    ) -> Any:
        """Execute with tracing."""
        # Start new trace if not provided
        if trace_context is None:
            trace_context = TraceContext.new_trace()
        
        print(f"[TRACE] {trace_context.trace_id[:8]} - {self.name} started")
        print(f"  Span: {trace_context.span_id[:8]}")
        if trace_context.parent_span_id:
            print(f"  Parent: {trace_context.parent_span_id[:8]}")
        
        # Do work
        result = await self._do_work(input, trace_context)
        
        print(f"[TRACE] {trace_context.trace_id[:8]} - {self.name} completed")
        
        return result
    
    async def _do_work(self, input: Any, trace_context: TraceContext) -> Any:
        """Work implementation."""
        # If calling another agent, pass child context
        return f"Result from {self.name}"

# Usage - trace flows through agents
trace = TraceContext.new_trace()

agent_a = TracedAgent("AgentA")
result_a = await agent_a.execute("task", trace)

# Agent A calls Agent B with child span
agent_b = TracedAgent("AgentB")
result_b = await agent_b.execute(result_a, trace.child_span())

# Can trace entire flow through system
\`\`\`

## Visualization Tools

\`\`\`python
class AgentVisualizer:
    """Visualize agent interactions."""
    
    def __init__(self, logger: AgentLogger):
        self.logger = logger
    
    def generate_sequence_diagram(self) -> str:
        """Generate mermaid sequence diagram."""
        diagram = ["sequenceDiagram"]
        
        for log in self.logger.logs:
            if log.error:
                diagram.append(
                    f"    {log.agent_name}->>X{log.agent_name}: {log.action} (ERROR)"
                )
            else:
                diagram.append(
                    f"    {log.agent_name}->>{log.agent_name}: {log.action}"
                )
        
        return "\\n".join(diagram)
    
    def generate_timeline_html(self) -> str:
        """Generate HTML timeline."""
        html = ["<div style='font-family: monospace;'>"]
        
        for log in self.logger.logs:
            color = "red" if log.error else "green"
            html.append(
                f"<div style='color: {color};'>"
                f"{log.timestamp:.2f}s: {log.agent_name}.{log.action} "
                f"({log.duration:.2f}s)"
                f"</div>"
            )
        
        html.append("</div>")
        return "".join(html)
    
    def print_tree(self):
        """Print execution tree."""
        print("Execution Tree:")
        print("==============")
        
        agent_actions = {}
        for log in self.logger.logs:
            if log.agent_name not in agent_actions:
                agent_actions[log.agent_name] = []
            agent_actions[log.agent_name].append(log)
        
        for agent_name, actions in agent_actions.items():
            print(f"\\n{agent_name}:")
            for action in actions:
                status = "❌" if action.error else "✅"
                print(f"  {status} {action.action} ({action.duration:.2f}s)")

# Usage
visualizer = AgentVisualizer(logger)
print(visualizer.generate_sequence_diagram())
visualizer.print_tree()
\`\`\`

## Performance Profiling

\`\`\`python
class AgentProfiler:
    """Profile agent performance."""
    
    def __init__(self, logger: AgentLogger):
        self.logger = logger
    
    def find_bottlenecks(self, threshold: float = 1.0) -> List[AgentLog]:
        """Find slow operations."""
        return [
            log for log in self.logger.logs
            if log.duration > threshold
        ]
    
    def get_slowest_agents(self, top_n: int = 5) -> List[tuple[str, float]]:
        """Get slowest agents by average duration."""
        agent_times = {}
        agent_counts = {}
        
        for log in self.logger.logs:
            if log.agent_name not in agent_times:
                agent_times[log.agent_name] = 0
                agent_counts[log.agent_name] = 0
            
            agent_times[log.agent_name] += log.duration
            agent_counts[log.agent_name] += 1
        
        averages = [
            (agent, agent_times[agent] / agent_counts[agent])
            for agent in agent_times
        ]
        
        averages.sort(key=lambda x: x[1], reverse=True)
        return averages[:top_n]
    
    def analyze_parallelism(self) -> Dict[str, Any]:
        """Analyze how well agents run in parallel."""
        if not self.logger.logs:
            return {}
        
        # Sort by start time
        sorted_logs = sorted(self.logger.logs, key=lambda l: l.timestamp)
        
        # Calculate total time span
        start = sorted_logs[0].timestamp
        end = max(l.timestamp + l.duration for l in sorted_logs)
        total_wall_time = end - start
        
        # Calculate total CPU time
        total_cpu_time = sum(l.duration for l in sorted_logs)
        
        # Parallelism efficiency
        parallelism = total_cpu_time / total_wall_time if total_wall_time > 0 else 0
        
        return {
            "total_wall_time": total_wall_time,
            "total_cpu_time": total_cpu_time,
            "parallelism_factor": parallelism,
            "efficiency": f"{(parallelism / len(set(l.agent_name for l in sorted_logs))) * 100:.1f}%"
        }

# Usage
profiler = AgentProfiler(logger)

# Find problems
bottlenecks = profiler.find_bottlenecks(threshold=0.5)
print(f"Found {len(bottlenecks)} bottlenecks")

slowest = profiler.get_slowest_agents(top_n=3)
print("Slowest agents:", slowest)

parallelism = profiler.analyze_parallelism()
print(f"Parallelism factor: {parallelism['parallelism_factor']:.2f}x")
\`\`\`

## State Inspection

\`\`\`python
class StateInspector:
    """Inspect system state for debugging."""
    
    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
        self.snapshots: List[Dict[str, Any]] = []
    
    def take_snapshot(self, label: str):
        """Take snapshot of current state."""
        snapshot = {
            "timestamp": time.time(),
            "label": label,
            "agent_states": {}
        }
        
        for name, agent in self.agents.items():
            if hasattr(agent, 'get_state'):
                snapshot["agent_states"][name] = agent.get_state()
        
        self.snapshots.append(snapshot)
    
    def compare_snapshots(
        self,
        snapshot1_idx: int,
        snapshot2_idx: int
    ) -> Dict[str, Any]:
        """Compare two snapshots."""
        snap1 = self.snapshots[snapshot1_idx]
        snap2 = self.snapshots[snapshot2_idx]
        
        differences = {}
        
        for agent_name in snap1["agent_states"]:
            if agent_name in snap2["agent_states"]:
                state1 = snap1["agent_states"][agent_name]
                state2 = snap2["agent_states"][agent_name]
                
                if state1 != state2:
                    differences[agent_name] = {
                        "before": state1,
                        "after": state2
                    }
        
        return differences
    
    def print_current_state(self):
        """Print current state of all agents."""
        print("Current State:")
        print("==============")
        
        for name, agent in self.agents.items():
            if hasattr(agent, 'get_state'):
                state = agent.get_state()
                print(f"\\n{name}:")
                print(f"  {state}")

# Usage
class StatefulAgent:
    def __init__(self, name: str):
        self.name = name
        self.state = {"tasks_completed": 0, "current_task": None}
    
    def get_state(self) -> Dict:
        return self.state.copy()

agents = {
    "agent1": StatefulAgent("Agent1"),
    "agent2": StatefulAgent("Agent2")
}

inspector = StateInspector(agents)

# Take snapshots at key points
inspector.take_snapshot("initial")
agents["agent1"].state["tasks_completed"] = 1
inspector.take_snapshot("after_task")

# Compare
diffs = inspector.compare_snapshots(0, 1)
print("State changes:", diffs)
\`\`\`

## Error Reproduction

\`\`\`python
class ErrorReproducer:
    """Reproduce errors from logs."""
    
    def __init__(self, logger: AgentLogger):
        self.logger = logger
    
    def get_error_context(self, error_log: AgentLog) -> Dict[str, Any]:
        """Get context around an error."""
        # Find logs before error
        error_idx = self.logger.logs.index(error_log)
        before = self.logger.logs[max(0, error_idx - 5):error_idx]
        
        return {
            "error": error_log,
            "preceding_actions": before,
            "agent_state": "..."  # Would capture state
        }
    
    async def replay_scenario(
        self,
        agents: Dict[str, Any],
        start_idx: int,
        end_idx: int
    ):
        """Replay sequence of actions."""
        print(f"Replaying actions {start_idx} to {end_idx}...")
        
        for log in self.logger.logs[start_idx:end_idx + 1]:
            agent = agents.get(log.agent_name)
            if agent:
                print(f"  Replaying: {log.agent_name}.{log.action}")
                try:
                    await agent.execute(log.input)
                except Exception as e:
                    print(f"    ❌ Error reproduced: {e}")
                    return
        
        print("  ✅ Replay completed successfully")

# Usage
# Find errors
errors = logger.get_logs(has_error=True)
if errors:
    reproducer = ErrorReproducer(logger)
    
    # Get context
    context = reproducer.get_error_context(errors[0])
    print("Error context:", context)
    
    # Try to reproduce
    # await reproducer.replay_scenario(agents, start_idx=0, end_idx=5)
\`\`\`

## Debugging Checklist

\`\`\`python
class DebugChecklists:
    """Common debugging checks."""
    
    @staticmethod
    def check_agent_issues(logger: AgentLogger) -> List[str]:
        """Check for common agent issues."""
        issues = []
        
        # Check for errors
        errors = logger.get_logs(has_error=True)
        if errors:
            issues.append(f"Found {len(errors)} errors")
        
        # Check for slow agents
        slow = [l for l in logger.logs if l.duration > 5.0]
        if slow:
            issues.append(f"Found {len(slow)} slow operations (>5s)")
        
        # Check for idle agents
        agent_actions = {}
        for log in logger.logs:
            agent_actions[log.agent_name] = agent_actions.get(log.agent_name, 0) + 1
        
        avg_actions = sum(agent_actions.values()) / len(agent_actions) if agent_actions else 0
        idle = [a for a, count in agent_actions.items() if count < avg_actions * 0.5]
        if idle:
            issues.append(f"Potentially idle agents: {idle}")
        
        return issues if issues else ["No issues found"]

# Usage
issues = DebugChecklist.check_agent_issues(logger)
for issue in issues:
    print(f"⚠️  {issue}")
\`\`\`

## Best Practices

1. **Log Everything**: Every agent action
2. **Use Trace IDs**: Track flows through system
3. **Snapshot State**: Capture state at key points
4. **Visualize**: Use diagrams and timelines
5. **Profile**: Find bottlenecks
6. **Reproduce**: Replay errors
7. **Automate Checks**: Run automated diagnostics

## Next Steps

You now understand debugging. Next, learn:
- Human-in-the-loop patterns
- Production deployment
- Monitoring and alerts
`,
};
