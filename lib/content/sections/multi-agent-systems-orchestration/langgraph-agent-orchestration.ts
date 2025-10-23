/**
 * LangGraph for Agent Orchestration Section
 * Module 7: Multi-Agent Systems & Orchestration
 */

export const langgraphagentorchestrationSection = {
  id: 'langgraph-agent-orchestration',
  title: 'LangGraph for Agent Orchestration',
  content: `# LangGraph for Agent Orchestration

Master using LangGraph to build sophisticated graph-based agent workflows.

## Overview: What is LangGraph?

LangGraph is a library for building stateful, multi-agent applications as graphs:

- **Graph-Based**: Define agents as nodes, flows as edges
- **Stateful**: Maintains state across agent executions
- **Cyclic**: Supports loops and iterations
- **Conditional**: Branch based on agent outputs
- **Streaming**: Stream results as they're generated

### Why LangGraph?

**vs Manual Coordination**: Less boilerplate  
**vs Simple Chains**: Supports complex flows  
**vs State Machines**: More flexible  
**Built for LLMs**: Designed for agent orchestration  

## Basic LangGraph Structure

\`\`\`python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
import operator

# 1. Define State
class AgentState(TypedDict):
    """State passed between agents."""
    messages: Annotated[list, operator.add]  # Accumulate messages
    current_task: str
    completed_tasks: list[str]
    final_output: str

# 2. Define Agent Nodes
async def research_node(state: AgentState) -> AgentState:
    """Research agent node."""
    print(f"[Research] Task: {state['current_task']}")
    
    # Simulate research
    result = f"Research results for {state['current_task']}"
    
    return {
        "messages": [{"role": "research", "content": result}],
        "completed_tasks": state['completed_tasks'] + ["research"]
    }

async def write_node(state: AgentState) -> AgentState:
    """Writer agent node."""
    print(f"[Write] Based on research")
    
    # Get research results
    research = next(
        (m["content"] for m in state["messages"] if m["role"] == "research"),
        ""
    )
    
    result = f"Article based on: {research[:50]}..."
    
    return {
        "messages": [{"role": "write", "content": result}],
        "completed_tasks": state['completed_tasks'] + ["write"]
    }

async def review_node(state: AgentState) -> AgentState:
    """Reviewer agent node."""
    print(f"[Review] Reviewing article")
    
    # Get article
    article = next(
        (m["content"] for m in state["messages"] if m["role"] == "write"),
        ""
    )
    
    result = f"Reviewed: {article[:50]}..."
    
    return {
        "messages": [{"role": "review", "content": result}],
        "completed_tasks": state['completed_tasks'] + ["review"],
        "final_output": result
    }

# 3. Build Graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("research", research_node)
workflow.add_node("write", write_node)
workflow.add_node("review", review_node)

# Add edges
workflow.add_edge("research", "write")
workflow.add_edge("write", "review")
workflow.add_edge("review", END)

# Set entry point
workflow.set_entry_point("research")

# Compile
app = workflow.compile()

# 4. Execute
initial_state = {
    "messages": [],
    "current_task": "quantum computing",
    "completed_tasks": [],
    "final_output": ""
}

result = await app.ainvoke(initial_state)
print(f"\\nFinal output: {result['final_output']}")
print(f"Completed: {result['completed_tasks']}")
\`\`\`

## Conditional Branching

\`\`\`python
from langgraph.graph import StateGraph, END

class ConditionalState(TypedDict):
    """State with quality tracking."""
    content: str
    quality_score: float
    iterations: int
    max_iterations: int

async def generate_node(state: ConditionalState) -> ConditionalState:
    """Generate content."""
    print(f"[Generate] Iteration {state['iterations'] + 1}")
    
    # Simulate generation
    content = f"Content v{state['iterations'] + 1}: ..."
    
    return {
        "content": content,
        "iterations": state['iterations'] + 1
    }

async def evaluate_node(state: ConditionalState) -> ConditionalState:
    """Evaluate quality."""
    print(f"[Evaluate] Checking quality...")
    
    # Simulate evaluation
    import random
    quality = min(1.0, 0.5 + state['iterations'] * 0.2 + random.random() * 0.1)
    
    return {
        "quality_score": quality
    }

def should_continue(state: ConditionalState) -> str:
    """Decide whether to continue or finish."""
    if state['quality_score'] >= 0.8:
        print("  Quality threshold reached!")
        return "end"
    elif state['iterations'] >= state['max_iterations']:
        print("  Max iterations reached!")
        return "end"
    else:
        print("  Quality not sufficient, regenerating...")
        return "continue"

# Build graph with conditional
workflow = StateGraph(ConditionalState)

workflow.add_node("generate", generate_node)
workflow.add_node("evaluate", evaluate_node)

# Add edges
workflow.add_edge("generate", "evaluate")

# Conditional edge from evaluate
workflow.add_conditional_edges(
    "evaluate",
    should_continue,
    {
        "continue": "generate",  # Loop back
        "end": END
    }
)

workflow.set_entry_point("generate")

app = workflow.compile()

# Execute
result = await app.ainvoke({
    "content": "",
    "quality_score": 0.0,
    "iterations": 0,
    "max_iterations": 5
})

print(f"\\nFinal quality: {result['quality_score']:.2f}")
print(f"Total iterations: {result['iterations']}")
\`\`\`

## Multi-Agent Collaboration

\`\`\`python
from typing import Literal

class CollaborationState(TypedDict):
    """State for multi-agent collaboration."""
    task: str
    research: str
    code: str
    tests: str
    review: str
    current_agent: str
    next_action: str

async def researcher(state: CollaborationState) -> CollaborationState:
    """Researcher agent."""
    print(f"[Researcher] Researching: {state['task']}")
    
    research = f"Research findings on {state['task']}: ..."
    
    return {
        "research": research,
        "current_agent": "researcher",
        "next_action": "code"
    }

async def coder(state: CollaborationState) -> CollaborationState:
    """Coder agent."""
    print(f"[Coder] Implementing based on research")
    
    code = f"""
def solution():
    # Based on: {state['research'][:30]}...
    pass
"""
    
    return {
        "code": code,
        "current_agent": "coder",
        "next_action": "test"
    }

async def tester(state: CollaborationState) -> CollaborationState:
    """Tester agent."""
    print(f"[Tester] Writing tests")
    
    tests = f"""
def test_solution():
    # Tests for: {state['code'][:30]}...
    pass
"""
    
    return {
        "tests": tests,
        "current_agent": "tester",
        "next_action": "review"
    }

async def reviewer(state: CollaborationState) -> CollaborationState:
    """Reviewer agent."""
    print(f"[Reviewer] Reviewing all work")
    
    review = "All components reviewed and approved"
    
    return {
        "review": review,
        "current_agent": "reviewer",
        "next_action": "end"
    }

def route_next(state: CollaborationState) -> Literal["code", "test", "review", "end"]:
    """Route to next agent."""
    action = state.get("next_action", "end")
    print(f"  Routing to: {action}")
    return action

# Build collaboration graph
workflow = StateGraph(CollaborationState)

workflow.add_node("research", researcher)
workflow.add_node("code", coder)
workflow.add_node("test", tester)
workflow.add_node("review", reviewer)

# Set up routing
workflow.set_entry_point("research")

workflow.add_conditional_edges(
    "research",
    route_next,
    {"code": "code", "end": END}
)

workflow.add_conditional_edges(
    "code",
    route_next,
    {"test": "test", "end": END}
)

workflow.add_conditional_edges(
    "test",
    route_next,
    {"review": "review", "end": END}
)

workflow.add_conditional_edges(
    "review",
    route_next,
    {"end": END}
)

app = workflow.compile()

# Execute
result = await app.ainvoke({
    "task": "Implement sorting algorithm",
    "research": "",
    "code": "",
    "tests": "",
    "review": "",
    "current_agent": "",
    "next_action": ""
})

print(f"\\nFinal review: {result['review']}")
\`\`\`

## Human-in-the-Loop

\`\`\`python
from langgraph.checkpoint.memory import MemorySaver

class HITLState(TypedDict):
    """State with human approval."""
    proposal: str
    approved: bool
    feedback: str
    iteration: int

async def propose_node(state: HITLState) -> HITLState:
    """Generate proposal."""
    print(f"[Propose] Creating proposal (iteration {state['iteration']})")
    
    proposal = f"Proposal v{state['iteration']}: ..."
    
    if state.get('feedback'):
        proposal += f" [Incorporating feedback: {state['feedback']}]"
    
    return {
        "proposal": proposal,
        "iteration": state['iteration'] + 1
    }

async def human_approval_node(state: HITLState) -> HITLState:
    """Wait for human approval."""
    print(f"\\nProposal: {state['proposal']}")
    print("Waiting for approval...")
    
    # In real system, this would pause and wait for human input
    # For demo, simulate
    import random
    approved = random.random() > 0.5
    
    if approved:
        print("✅ Approved!")
        return {"approved": True, "feedback": ""}
    else:
        print("❌ Needs revision")
        return {"approved": False, "feedback": "Add more detail"}

def check_approval(state: HITLState) -> str:
    """Check if approved."""
    return "approved" if state.get('approved', False) else "revise"

# Build HITL graph
workflow = StateGraph(HITLState)

workflow.add_node("propose", propose_node)
workflow.add_node("human_approval", human_approval_node)

workflow.set_entry_point("propose")
workflow.add_edge("propose", "human_approval")

workflow.add_conditional_edges(
    "human_approval",
    check_approval,
    {
        "approved": END,
        "revise": "propose"
    }
)

# Use checkpointer to save state between human interactions
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# Execute
result = await app.ainvoke(
    {
        "proposal": "",
        "approved": False,
        "feedback": "",
        "iteration": 1
    },
    config={"configurable": {"thread_id": "demo"}}
)

print(f"\\nFinal proposal: {result['proposal']}")
\`\`\`

## Streaming Results

\`\`\`python
class StreamingState(TypedDict):
    """State for streaming."""
    input: str
    outputs: list[str]

async def stream_node_1(state: StreamingState) -> StreamingState:
    """First processing node."""
    print("[Node 1] Processing...")
    await asyncio.sleep(0.5)
    return {
        "outputs": state['outputs'] + ["Output from node 1"]
    }

async def stream_node_2(state: StreamingState) -> StreamingState:
    """Second processing node."""
    print("[Node 2] Processing...")
    await asyncio.sleep(0.5)
    return {
        "outputs": state['outputs'] + ["Output from node 2"]
    }

async def stream_node_3(state: StreamingState) -> StreamingState:
    """Third processing node."""
    print("[Node 3] Processing...")
    await asyncio.sleep(0.5)
    return {
        "outputs": state['outputs'] + ["Output from node 3"]
    }

# Build graph
workflow = StateGraph(StreamingState)

workflow.add_node("node1", stream_node_1)
workflow.add_node("node2", stream_node_2)
workflow.add_node("node3", stream_node_3)

workflow.set_entry_point("node1")
workflow.add_edge("node1", "node2")
workflow.add_edge("node2", "node3")
workflow.add_edge("node3", END)

app = workflow.compile()

# Stream results as they come
print("Streaming results:")
async for chunk in app.astream({"input": "test", "outputs": []}):
    print(f"  Chunk: {chunk}")
\`\`\`

## Parallel Execution in LangGraph

\`\`\`python
from langgraph.graph import StateGraph, END

class ParallelState(TypedDict):
    """State for parallel execution."""
    input: str
    analysis_a: str
    analysis_b: str
    analysis_c: str
    combined: str

async def analyzer_a(state: ParallelState) -> ParallelState:
    """First analyzer."""
    print("[Analyzer A] Analyzing...")
    await asyncio.sleep(1)
    return {"analysis_a": "Analysis A results"}

async def analyzer_b(state: ParallelState) -> ParallelState:
    """Second analyzer."""
    print("[Analyzer B] Analyzing...")
    await asyncio.sleep(1.5)
    return {"analysis_b": "Analysis B results"}

async def analyzer_c(state: ParallelState) -> ParallelState:
    """Third analyzer."""
    print("[Analyzer C] Analyzing...")
    await asyncio.sleep(0.8)
    return {"analysis_c": "Analysis C results"}

async def combiner(state: ParallelState) -> ParallelState:
    """Combine all analyses."""
    print("[Combiner] Combining results...")
    
    combined = f"""
Combined Analysis:
- A: {state['analysis_a']}
- B: {state['analysis_b']}
- C: {state['analysis_c']}
"""
    
    return {"combined": combined}

# Build graph with parallel execution
workflow = StateGraph(ParallelState)

workflow.add_node("analyzer_a", analyzer_a)
workflow.add_node("analyzer_b", analyzer_b)
workflow.add_node("analyzer_c", analyzer_c)
workflow.add_node("combiner", combiner)

workflow.set_entry_point("analyzer_a")

# All analyzers run in parallel
workflow.add_edge("analyzer_a", "combiner")
workflow.add_edge("analyzer_b", "combiner")
workflow.add_edge("analyzer_c", "combiner")

# Need to start all parallel nodes
workflow.add_edge("__start__", "analyzer_a")
workflow.add_edge("__start__", "analyzer_b")
workflow.add_edge("__start__", "analyzer_c")

workflow.add_edge("combiner", END)

app = workflow.compile()

import time
start = time.time()

result = await app.ainvoke({
    "input": "test",
    "analysis_a": "",
    "analysis_b": "",
    "analysis_c": "",
    "combined": ""
})

elapsed = time.time() - start
print(f"\\nCompleted in {elapsed:.2f}s")  # ~1.5s (longest), not 3.3s (sum)
print(result['combined'])
\`\`\`

## Error Handling

\`\`\`python
class ErrorHandlingState(TypedDict):
    """State with error tracking."""
    data: str
    error: Optional[str]
    retry_count: int
    max_retries: int

async def risky_node(state: ErrorHandlingState) -> ErrorHandlingState:
    """Node that might fail."""
    print(f"[Risky] Attempt {state['retry_count'] + 1}")
    
    import random
    if random.random() < 0.5 and state['retry_count'] < state['max_retries']:
        # Simulate failure
        return {
            "error": "Random failure occurred",
            "retry_count": state['retry_count'] + 1
        }
    else:
        # Success
        return {
            "data": "Processed data",
            "error": None,
            "retry_count": state['retry_count'] + 1
        }

def check_error(state: ErrorHandlingState) -> str:
    """Check if error occurred."""
    if state.get('error') and state['retry_count'] < state['max_retries']:
        print(f"  Error: {state['error']}, retrying...")
        return "retry"
    elif state.get('error'):
        print(f"  Max retries reached!")
        return "failed"
    else:
        return "success"

# Build graph with error handling
workflow = StateGraph(ErrorHandlingState)

workflow.add_node("risky", risky_node)

workflow.set_entry_point("risky")

workflow.add_conditional_edges(
    "risky",
    check_error,
    {
        "retry": "risky",  # Loop back to retry
        "success": END,
        "failed": END
    }
)

app = workflow.compile()

result = await app.ainvoke({
    "data": "",
    "error": None,
    "retry_count": 0,
    "max_retries": 3
})

if result.get('error'):
    print(f"\\n❌ Failed: {result['error']}")
else:
    print(f"\\n✅ Success: {result['data']}")
\`\`\`

## Best Practices

1. **Define State Clearly**: Type your state for better errors
2. **Use Operators**: Annotate how fields combine (add, override)
3. **Keep Nodes Focused**: Each node does one thing
4. **Handle Errors**: Add error states and recovery
5. **Use Checkpointing**: For long-running workflows
6. **Stream When Possible**: Better UX
7. **Test Graphs**: Unit test individual nodes

## Real-World Example: Research Assistant

\`\`\`python
class ResearchState(TypedDict):
    """Complete research workflow state."""
    query: str
    search_results: list[str]
    analysis: str
    draft_report: str
    review_feedback: str
    final_report: str
    quality_score: float
    iteration: int

async def search_node(state: ResearchState) -> ResearchState:
    """Search for information."""
    # Use web search API
    results = ["Result 1", "Result 2", "Result 3"]
    return {"search_results": results}

async def analyze_node(state: ResearchState) -> ResearchState:
    """Analyze search results."""
    analysis = "Analysis of search results..."
    return {"analysis": analysis}

async def draft_node(state: ResearchState) -> ResearchState:
    """Draft report."""
    draft = f"Report based on: {state['analysis'][:50]}..."
    return {"draft_report": draft, "iteration": state['iteration'] + 1}

async def review_node(state: ResearchState) -> ResearchState:
    """Review report quality."""
    import random
    quality = min(1.0, 0.6 + state['iteration'] * 0.15 + random.random() * 0.1)
    feedback = "Add more detail" if quality < 0.8 else "Looks good"
    return {
        "quality_score": quality,
        "review_feedback": feedback
    }

def should_revise(state: ResearchState) -> str:
    """Decide if revision needed."""
    if state['quality_score'] >= 0.8:
        return "finalize"
    elif state['iteration'] >= 3:
        return "finalize"
    else:
        return "revise"

async def finalize_node(state: ResearchState) -> ResearchState:
    """Finalize report."""
    return {"final_report": state['draft_report']}

# Build complete workflow
workflow = StateGraph(ResearchState)

workflow.add_node("search", search_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("draft", draft_node)
workflow.add_node("review", review_node)
workflow.add_node("finalize", finalize_node)

workflow.set_entry_point("search")
workflow.add_edge("search", "analyze")
workflow.add_edge("analyze", "draft")
workflow.add_edge("draft", "review")

workflow.add_conditional_edges(
    "review",
    should_revise,
    {
        "revise": "draft",
        "finalize": "finalize"
    }
)

workflow.add_edge("finalize", END)

app = workflow.compile()

# Execute
result = await app.ainvoke({
    "query": "How does LangGraph work?",
    "search_results": [],
    "analysis": "",
    "draft_report": "",
    "review_feedback": "",
    "final_report": "",
    "quality_score": 0.0,
    "iteration": 0
})

print(f"Final Report: {result['final_report']}")
print(f"Quality: {result['quality_score']:.2f}")
print(f"Iterations: {result['iteration']}")
\`\`\`

## Next Steps

You now understand LangGraph. Next, learn:
- Other frameworks (CrewAI)
- Debugging multi-agent systems
- Production deployment
`,
};
