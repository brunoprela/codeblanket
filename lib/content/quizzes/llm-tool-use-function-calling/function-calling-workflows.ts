export const functionCallingWorkflowsQuiz = [
  {
    id: 'q1',
    question:
      'Design a workflow orchestration system that can handle conditional logic, loops, and parallel execution of function calls. How would you represent the workflow, manage state between steps, and handle errors at each stage?',
    sampleAnswer: `A sophisticated workflow orchestration system needs to handle complex control flow while maintaining clarity and reliability.

**Workflow Representation:**

Use a DAG (Directed Acyclic Graph) with node types for different operations:

\`\`\`python
class WorkflowNode:
    type: str  # "tool", "condition", "parallel", "loop"
    tool_name: str = None
    condition: Callable = None
    children: List[WorkflowNode] = []
    
class Workflow:
    start_node: WorkflowNode
    state: Dict[str, Any] = {}
    
# Example workflow
workflow = Workflow(
    start_node=ToolNode("search_location", 
        children=[
            ConditionalNode(
                condition=lambda state: state['location_found'],
                if_true=ToolNode("get_weather"),
                if_false=ToolNode("ask_user_clarification")
            )
        ]
    )
)
\`\`\`

**State Management:**

Maintain shared state across workflow steps with versioning:

\`\`\`python
class WorkflowState:
    def __init__(self):
        self.data = {}
        self.history = []
        
    def set(self, key, value):
        self.data[key] = value
        self.history.append({"key": key, "value": value, "timestamp": now()})
    
    def get(self, key, default=None):
        return self.data.get(key, default)
    
    def checkpoint(self):
        return deepcopy(self.data)
    
    def rollback(self, checkpoint):
        self.data = checkpoint
\`\`\`

**Execution Engine:**

\`\`\`python
class WorkflowExecutor:
    async def execute(self, workflow: Workflow) -> Dict:
        state = WorkflowState()
        results = []
        
        try:
            result = await self._execute_node(workflow.start_node, state)
            return {"status": "success", "result": result, "state": state.data}
        except Exception as e:
            return {"status": "error", "error": str(e), "state": state.data}
    
    async def _execute_node(self, node: WorkflowNode, state: WorkflowState):
        if node.type == "tool":
            return await self._execute_tool(node, state)
        elif node.type == "condition":
            return await self._execute_conditional(node, state)
        elif node.type == "parallel":
            return await self._execute_parallel(node, state)
        elif node.type == "loop":
            return await self._execute_loop(node, state)
\`\`\`

**Error Handling:**

Multi-level error handling with recovery:

\`\`\`python
class ErrorHandler:
    def __init__(self):
        self.max_retries = 3
        self.fallback_strategies = {}
    
    async def handle_tool_error(self, node, state, error):
        # Retry with exponential backoff
        for attempt in range(self.max_retries):
            try:
                return await execute_with_retry(node, state)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    # Try fallback
                    if node.tool_name in self.fallback_strategies:
                        return await self.fallback_strategies[node.tool_name](state)
                    raise
\`\`\`

**Parallel Execution:**

\`\`\`python
async def _execute_parallel(self, node: WorkflowNode, state: WorkflowState):
    tasks = [self._execute_node(child, state) for child in node.children]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle partial failures
    successful = [r for r in results if not isinstance(r, Exception)]
    failed = [r for r in results if isinstance(r, Exception)]
    
    if failed and not successful:
        raise Exception("All parallel tasks failed")
    
    return {"successful": successful, "failed": len(failed)}
\`\`\`

**Loop Implementation:**

\`\`\`python
async def _execute_loop(self, node: WorkflowNode, state: WorkflowState):
    max_iterations = node.max_iterations or 10
    results = []
    
    for i in range(max_iterations):
        if node.condition(state):
            break
        
        result = await self._execute_node(node.body, state)
        results.append(result)
    
    return results
\`\`\`

**Best Practices:**
- Implement timeouts at each level
- Save checkpoints for long workflows
- Provide progress updates
- Allow workflow cancellation
- Log every step for debugging
- Support workflow visualization`,
    keyPoints: [
      'Use DAG representation for workflow structure',
      'Implement robust state management and error handling',
      'Support conditional logic, loops, and parallel execution',
      'Provide monitoring and visualization capabilities',
    ],
  },
  {
    id: 'q2',
    question:
      'Explain how you would implement the ReAct (Reasoning + Acting) pattern for complex problem-solving tasks. What are the advantages over simple sequential execution, and how do you prevent the agent from getting stuck in loops?',
    sampleAnswer: `The ReAct pattern alternates between reasoning about what to do next and taking actions, leading to more robust and explainable agent behavior.

**ReAct Implementation:**

\`\`\`python
class ReActAgent:
    def __init__(self, tools, max_steps=10):
        self.tools = tools
        self.max_steps = max_steps
        self.thoughts = []
        self.actions = []
    
    async def solve(self, problem: str):
        for step in range(self.max_steps):
            # THINK: Reason about next action
            thought = await self._think(problem, self.history())
            self.thoughts.append(thought)
            
            # Check if done
            if self._is_complete(thought):
                return self._final_answer(thought)
            
            # ACT: Execute chosen action
            action = self._parse_action(thought)
            if action:
                observation = await self._execute_action(action)
                self.actions.append({"action": action, "observation": observation})
            
            # Detect loops
            if self._detect_loop():
                thought = await self._break_loop()
        
        return "Could not solve within step limit"
    
    async def _think(self, problem: str, history: str):
        prompt = f"""Problem: {problem}

History:
{history}

Think step by step:
1. What do I know so far?
2. What do I still need to find out?
3. What action should I take next?
4. Or am I ready to answer?

Thought:"""
        
        response = await llm_call(prompt)
        return response
    
    def history(self):
        history_parts = []
        for i, (thought, action) in enumerate(zip(self.thoughts, self.actions)):
            history_parts.append(f"Step {i+1}:")
            history_parts.append(f"Thought: {thought}")
            history_parts.append(f"Action: {action['action']}")
            history_parts.append(f"Observation: {action['observation']}")
        return "\\n\\n".join(history_parts)
\`\`\`

**Advantages over Sequential:**

1. **Explicit Reasoning**: Thoughts are recorded and auditable
2. **Adaptive**: Can change strategy based on observations
3. **Error Recovery**: Can reason about failures and try alternatives
4. **Explainable**: Clear chain of reasoning for debugging
5. **Human-Like**: Mimics how humans solve problems

**Loop Detection and Prevention:**

\`\`\`python
def _detect_loop(self):
    if len(self.actions) < 3:
        return False
    
    # Check for repeated actions
    recent_actions = self.actions[-3:]
    action_signatures = [self._action_signature(a) for a in recent_actions]
    
    if len(set(action_signatures)) == 1:
        # Same action repeated 3 times
        return True
    
    # Check for oscillation (A -> B -> A -> B)
    if len(self.actions) >= 4:
        if action_signatures[-4] == action_signatures[-2] and \
           action_signatures[-3] == action_signatures[-1]:
            return True
    
    return False

def _action_signature(self, action):
    return f"{action['action']['name']}:{action['action']['args']}"

async def _break_loop(self):
    # Force different approach
    prompt = f"""You seem to be in a loop. Your last actions were:
{self.recent_actions_summary()}

Think of a DIFFERENT approach. What else could you try?"""
    
    new_thought = await llm_call(prompt)
    return new_thought
\`\`\`

**Memory and Context:**

\`\`\`python
class ReActMemory:
    def __init__(self, max_items=20):
        self.items = []
        self.max_items = max_items
    
    def add(self, thought, action, observation):
        self.items.append({
            "thought": thought,
            "action": action,
            "observation": observation,
            "timestamp": now()
        })
        
        # Keep only recent items to stay within context window
        if len(self.items) > self.max_items:
            # Summarize old items
            summary = self._summarize(self.items[:-10])
            self.items = [{"summary": summary}] + self.items[-10:]
\`\`\`

**Stopping Conditions:**

\`\`\`python
def _is_complete(self, thought: str) -> bool:
    # Look for completion signals
    completion_phrases = [
        "final answer",
        "i have enough information",
        "conclusion:",
        "therefore,"
    ]
    
    thought_lower = thought.lower()
    return any(phrase in thought_lower for phrase in completion_phrases)
\`\`\`

**Example Execution:**

\`\`\`
Problem: "What's the population of the capital of France?"

Step 1:
Thought: I need to first identify the capital of France, which I know is Paris. Then I need to find its population.
Action: search_web("population of Paris")
Observation: "Paris has a population of 2.2 million (city) and 12.5 million (metro)"

Step 2:
Thought: I found the population information. The city proper has 2.2 million, and the metropolitan area has 12.5 million. I should provide both numbers for completeness.
Action: None (ready for final answer)

Final Answer: The capital of France is Paris, which has a population of approximately 2.2 million in the city proper and 12.5 million in the metropolitan area.
\`\`\`

**Best Practices:**
- Limit maximum steps (typically 10-15)
- Implement loop detection early
- Summarize long histories to fit context
- Make thoughts explicit and structured
- Log everything for debugging
- Have escape hatches for stuck states
- Test with various problem types`,
    keyPoints: [
      'ReAct pattern alternates between reasoning and acting',
      'Implement loop detection and maximum step limits',
      'Use observation summaries to manage context length',
      'Make reasoning explicit for transparency and debugging',
    ],
  },
  {
    id: 'q3',
    question:
      'Compare and contrast different workflow orchestration frameworks (LangChain, LangGraph, CrewAI). For what types of applications would you choose each, and how would you migrate from one to another if requirements change?',
    sampleAnswer: `Different frameworks have different strengths and ideal use cases:

**LangChain:**

Strengths:
- Mature ecosystem with many integrations
- Extensive documentation and community
- Chains and agents abstractions
- Good for rapid prototyping
- Many pre-built components

Weaknesses:
- Can be overly abstracted
- Performance overhead
- Harder to debug complex chains
- Opinionated structure

Best for:
- Quick prototypes
- Standard RAG applications
- Simple sequential workflows
- Teams new to LLM applications

**LangGraph:**

Strengths:
- Graph-based workflow representation
- Explicit state management
- Conditional branching
- Cycles and loops supported
- Good for complex flows
- Easier to visualize

Weaknesses:
- Newer, smaller community
- More code needed for simple cases
- Steeper learning curve

Best for:
- Complex multi-step workflows
- Conditional logic and loops
- State machines
- Applications needing auditability
- When you need full control

**CrewAI:**

Strengths:
- Multi-agent collaboration focus
- Role-based agent design
- Task delegation patterns
- Good for team-like behavior
- Higher-level abstractions

Weaknesses:
- Less flexible than LangGraph
- Opinionated agent structure
- Smaller ecosystem

Best for:
- Multi-agent systems
- Role-based task distribution
- Simulating team collaboration
- Business process automation

**Custom Implementation:**

Strengths:
- Full control
- Optimized for your use case
- No framework overhead
- Easier debugging

Weaknesses:
- More code to write
- Need to reinvent patterns
- Maintenance burden

Best for:
- Production systems with specific needs
- Performance-critical applications
- When frameworks don't fit

**Decision Matrix:**

Simple RAG → LangChain
Complex workflows → LangGraph  
Multi-agent collaboration → CrewAI
Unique requirements → Custom

**Migration Strategy:**

1. **Decouple Your Logic:**
\`\`\`python
# Don't do this (framework-coupled):
from langchain import Chain
class MyApp(Chain):
    def _call(self, inputs):
        return self.llm(inputs)

# Do this (framework-agnostic):
class MyAppLogic:
    def process(self, inputs):
        return call_llm(inputs)

# Framework adapter
class LangChainAdapter:
    def __init__(self, logic):
        self.logic = logic
\`\`\`

2. **Abstract Your Tools:**
\`\`\`python
# Universal tool interface
class Tool:
    def execute(self, **kwargs):
        pass

# Framework-specific wrappers
class LangChainTool(BaseTool):
    def __init__(self, universal_tool):
        self.tool = universal_tool
    
    def _run(self, **kwargs):
        return self.tool.execute(**kwargs)
\`\`\`

3. **Migration Path:**
\`\`\`python
# Phase 1: Wrapper pattern
langgraph_executor = LangGraphWrapper(langchain_app)

# Phase 2: Parallel execution
results_old = langchain_app.run(input)
results_new = langgraph_app.run(input)
assert_equivalent(results_old, results_new)

# Phase 3: Gradual cutover
if feature_flag("use_langgraph"):
    return langgraph_app.run(input)
else:
    return langchain_app.run(input)
\`\`\`

**Practical Example:**

\`\`\`python
# Your core logic (framework-agnostic)
class ResearchAgent:
    def __init__(self, tools):
        self.tools = tools
    
    async def research(self, topic):
        # Your logic here
        pass

# LangChain adapter
from langchain.agents import Agent
class LangChainResearchAgent(Agent):
    def __init__(self, core_agent):
        self.agent = core_agent
    
    def _call(self, inputs):
        return await self.agent.research(inputs["topic"])

# LangGraph adapter  
from langgraph.graph import StateGraph
def create_langgraph_agent(core_agent):
    workflow = StateGraph(AgentState)
    workflow.add_node("research", lambda state: core_agent.research(state["topic"]))
    return workflow.compile()

# CrewAI adapter
from crewai import Agent as CrewAgent
class CrewAIResearchAgent(CrewAgent):
    def __init__(self, core_agent):
        super().__init__(role="researcher", goal="Research topics")
        self.core_agent = core_agent
\`\`\`

**Key Insights:**
- Keep business logic separate from framework
- Use adapters for framework integration
- Test with multiple frameworks early
- Have migration plan before committing
- Monitor performance across frameworks
- Document framework-specific quirks`,
    keyPoints: [
      'Different frameworks have different strengths and trade-offs',
      'Choose based on project needs, team experience, and requirements',
      'Keep business logic separate from framework for easier migration',
      'Use adapter pattern to decouple core logic from framework',
    ],
  },
];
