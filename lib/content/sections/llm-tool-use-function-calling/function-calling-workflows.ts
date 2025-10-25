export const functionCallingWorkflows = {
  title: 'Function Calling Workflows',
  id: 'function-calling-workflows',
  description:
    'Master different workflow patterns for orchestrating function calls, including sequential, parallel, conditional, and looping patterns.',
  content: `

# Function Calling Workflows

## Introduction

Real-world applications rarely involve just a single function call. Most tasks require multiple function calls orchestrated in sophisticated patterns: sequential execution, parallel calls, conditional branching, loops, and error recovery. Mastering these workflow patterns is essential for building production-ready agentic systems.

In this section, we'll explore the different workflow patterns, when to use each, and how to implement them robustly. We'll see how systems like ChatGPT orchestrate complex multi-step tasks using these patterns.

## Single Function Call Pattern

The simplest pattern: user request → single function call → response.

\`\`\`python
import openai
import json

def simple_function_call (user_message: str, functions: list, function_registry: dict):
    """
    Handle a simple single function call.
    """
    # Initial LLM call
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": user_message}],
        functions=functions
    )
    
    message = response.choices[0].message
    
    # Execute function if called
    if message.function_call:
        func_name = message.function_call.name
        func_args = json.loads (message.function_call.arguments)
        
        # Execute
        result = function_registry[func_name](**func_args)
        
        # Get final response
        final_response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": user_message},
                message.to_dict(),
                {
                    "role": "function",
                    "name": func_name,
                    "content": json.dumps (result)
                }
            ]
        )
        
        return final_response.choices[0].message.content
    else:
        return message.content

# Example
result = simple_function_call(
    "What\'s the weather in San Francisco?",
    functions=[WEATHER_FUNCTION],
    function_registry={"get_weather": get_weather}
)
\`\`\`

**Use case**: Simple queries that need one piece of information.

## Sequential Function Calls

Multiple function calls executed one after another, where each call may depend on previous results.

\`\`\`python
def sequential_function_calls (user_message: str, functions: list, 
                             function_registry: dict, max_iterations: int = 5):
    """
    Handle sequential function calls in a loop.
    """
    messages = [{"role": "user", "content": user_message}]
    
    for iteration in range (max_iterations):
        print(f"\\n--- Iteration {iteration + 1} ---")
        
        # Call LLM
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            functions=functions
        )
        
        message = response.choices[0].message
        messages.append (message.to_dict())
        
        # Check if function call
        if message.function_call:
            func_name = message.function_call.name
            func_args = json.loads (message.function_call.arguments)
            
            print(f"Calling: {func_name}({func_args})")
            
            # Execute function
            try:
                result = function_registry[func_name](**func_args)
                print(f"Result: {result}")
                
                # Add function result to messages
                messages.append({
                    "role": "function",
                    "name": func_name,
                    "content": json.dumps (result)
                })
            except Exception as e:
                messages.append({
                    "role": "function",
                    "name": func_name,
                    "content": json.dumps({"error": str (e)})
                })
        else:
            # No function call = final answer
            return message.content
    
    return "Max iterations reached"

# Example: Multi-step task
result = sequential_function_calls(
    "What's the weather in SF and NYC? Which is warmer?",
    functions=[WEATHER_FUNCTION],
    function_registry={"get_weather": get_weather}
)
\`\`\`

**Use case**: Tasks requiring multiple sequential steps, like "get data, process it, then get more data based on results."

### Sequential with Dependencies

\`\`\`python
# Example: Book a restaurant
# 1. Search restaurants
# 2. Get details of top choice
# 3. Check availability
# 4. Make reservation

user: "Book a table for 2 at an Italian restaurant in SF tonight at 7pm"
  ↓
LLM: search_restaurants (cuisine="Italian", location="SF", party_size=2)
  ↓
Result: [{"id": "rest_123", "name": "Pasta Place", "rating": 4.5}, ...]
  ↓
LLM: get_restaurant_details (restaurant_id="rest_123")
  ↓
Result: {"name": "Pasta Place", "address": "...", "phone": "..."}
  ↓
LLM: check_availability (restaurant_id="rest_123", date="2024-01-15", time="19:00", party_size=2)
  ↓
Result: {"available": true, "available_times": ["18:30", "19:00", "19:30"]}
  ↓
LLM: make_reservation (restaurant_id="rest_123", date="2024-01-15", time="19:00", party_size=2, name="John Doe")
  ↓
Result: {"confirmation": "RES-12345", "status": "confirmed"}
  ↓
LLM: "I've booked a table for 2 at Pasta Place tonight at 7pm. Your confirmation number is RES-12345."
\`\`\`

## Parallel Function Calls

Execute multiple independent function calls simultaneously for efficiency.

\`\`\`python
import asyncio
from typing import List, Dict, Any

async def execute_function_async (func, args):
    """Execute a function asynchronously."""
    # If function is already async
    if asyncio.iscoroutinefunction (func):
        return await func(**args)
    # If function is synchronous, run in executor
    else:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(**args))

async def parallel_function_calls (function_calls: List[Dict[str, Any]], 
                                  function_registry: dict):
    """
    Execute multiple function calls in parallel.
    
    function_calls: [
        {"name": "get_weather", "arguments": {"location": "SF"}},
        {"name": "get_weather", "arguments": {"location": "NYC"}}
    ]
    """
    # Create tasks for all function calls
    tasks = []
    for call in function_calls:
        func_name = call["name"]
        func_args = call["arguments"]
        func = function_registry[func_name]
        
        task = execute_function_async (func, func_args)
        tasks.append((func_name, func_args, task))
    
    # Wait for all to complete
    results = []
    for func_name, func_args, task in tasks:
        try:
            result = await task
            results.append({
                "function": func_name,
                "arguments": func_args,
                "result": result,
                "success": True
            })
        except Exception as e:
            results.append({
                "function": func_name,
                "arguments": func_args,
                "error": str (e),
                "success": False
            })
    
    return results

# Example usage
async def main():
    calls = [
        {"name": "get_weather", "arguments": {"location": "San Francisco"}},
        {"name": "get_weather", "arguments": {"location": "New York"}},
        {"name": "get_weather", "arguments": {"location": "London"}}
    ]
    
    results = await parallel_function_calls (calls, function_registry)
    
    for r in results:
        print(f"{r['function']}: {r['result']}")

asyncio.run (main())
\`\`\`

**Use case**: Independent queries that don't depend on each other's results. Much faster than sequential.

### GPT-4 Turbo Parallel Function Calling

GPT-4 Turbo supports native parallel function calling:

\`\`\`python
response = openai.chat.completions.create(
    model="gpt-4-1106-preview",  # Turbo model
    messages=[{
        "role": "user",
        "content": "What\'s the weather in SF, NYC, and London?"
    }],
    functions=[WEATHER_FUNCTION],
    function_call="auto"
)

# Response may contain multiple function calls
message = response.choices[0].message

# In GPT-4 Turbo, message.function_calls (plural) contains list
if hasattr (message, 'function_calls') and message.function_calls:
    for function_call in message.function_calls:
        func_name = function_call.name
        func_args = json.loads (function_call.arguments)
        print(f"Call: {func_name}({func_args})")
\`\`\`

## Conditional Function Calls

Execute different functions based on conditions or previous results.

\`\`\`python
def conditional_workflow (user_message: str):
    """
    Workflow with conditional logic.
    Example: Check if user exists, if yes get profile, if no create user.
    """
    messages = [{"role": "user", "content": user_message}]
    
    functions = [
        USER_EXISTS_FUNCTION,
        GET_USER_FUNCTION,
        CREATE_USER_FUNCTION
    ]
    
    # Step 1: Check if user exists
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        functions=functions
    )
    
    message = response.choices[0].message
    messages.append (message.to_dict())
    
    if message.function_call and message.function_call.name == "user_exists":
        args = json.loads (message.function_call.arguments)
        exists = user_exists(**args)
        
        messages.append({
            "role": "function",
            "name": "user_exists",
            "content": json.dumps({"exists": exists})
        })
        
        # Step 2: Conditional logic
        if exists:
            # Get existing user
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=messages,
                functions=[GET_USER_FUNCTION]
            )
        else:
            # Create new user
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=messages,
                functions=[CREATE_USER_FUNCTION]
            )
        
        # Continue processing...
        message = response.choices[0].message
        if message.function_call:
            # Execute the chosen function
            func_name = message.function_call.name
            func_args = json.loads (message.function_call.arguments)
            result = function_registry[func_name](**func_args)
            
            messages.append (message.to_dict())
            messages.append({
                "role": "function",
                "name": func_name,
                "content": json.dumps (result)
            })
        
        # Get final response
        final_response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        
        return final_response.choices[0].message.content
\`\`\`

**Use case**: Workflows with branching logic, error handling paths, or user-specific flows.

## Loop/Iterative Patterns

Keep calling functions until a condition is met.

\`\`\`python
def iterative_search_workflow (query: str, max_attempts: int = 3):
    """
    Iteratively refine search until satisfactory results found.
    """
    messages = [{"role": "user", "content": query}]
    
    for attempt in range (max_attempts):
        print(f"\\nAttempt {attempt + 1}")
        
        # Search
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            functions=[SEARCH_FUNCTION, REFINE_SEARCH_FUNCTION]
        )
        
        message = response.choices[0].message
        messages.append (message.to_dict())
        
        if message.function_call:
            func_name = message.function_call.name
            func_args = json.loads (message.function_call.arguments)
            
            if func_name == "search":
                results = search(**func_args)
                print(f"Found {len (results)} results")
                
                messages.append({
                    "role": "function",
                    "name": func_name,
                    "content": json.dumps({"results": results, "count": len (results)})
                })
                
                # Check if results are sufficient
                if len (results) >= 5:
                    # Enough results, get final answer
                    final_response = openai.chat.completions.create(
                        model="gpt-4",
                        messages=messages
                    )
                    return final_response.choices[0].message.content
                else:
                    # Ask LLM to refine search
                    messages.append({
                        "role": "system",
                        "content": "Not enough results. Try refining the search query."
                    })
            
            elif func_name == "refine_search":
                # LLM is refining the search
                print(f"Refining search with: {func_args}")
                continue
        else:
            # No function call = final answer (even with few results)
            return message.content
    
    return "Could not find satisfactory results after max attempts"
\`\`\`

**Use case**: Search refinement, iterative data processing, progressive disclosure.

## ReAct Pattern (Reason + Act)

Alternate between reasoning and acting (function calling).

\`\`\`python
def react_agent (user_goal: str, functions: list, function_registry: dict, 
                max_steps: int = 10):
    """
    ReAct agent: Reason about what to do, then Act (call function).
    """
    messages = [
        {
            "role": "system",
            "content": """You are a helpful agent that solves problems step by step.
            
For each step:
1. THINK: Reason about what you need to do next
2. ACT: Call a function to take action
3. OBSERVE: Analyze the result

Format your thoughts explicitly before taking action."""
        },
        {
            "role": "user",
            "content": user_goal
        }
    ]
    
    for step in range (max_steps):
        print(f"\\n--- Step {step + 1} ---")
        
        # Get LLM response
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            functions=functions
        )
        
        message = response.choices[0].message
        
        # THINK: Display reasoning
        if message.content:
            print(f"THINK: {message.content}")
        
        messages.append (message.to_dict())
        
        # ACT: Execute function if called
        if message.function_call:
            func_name = message.function_call.name
            func_args = json.loads (message.function_call.arguments)
            
            print(f"ACT: {func_name}({func_args})")
            
            try:
                result = function_registry[func_name](**func_args)
                print(f"OBSERVE: {result}")
                
                messages.append({
                    "role": "function",
                    "name": func_name,
                    "content": json.dumps (result)
                })
            except Exception as e:
                print(f"OBSERVE: Error - {e}")
                messages.append({
                    "role": "function",
                    "name": func_name,
                    "content": json.dumps({"error": str (e)})
                })
        else:
            # No function call = final answer
            print(f"\\nFINAL ANSWER: {message.content}")
            return message.content
    
    return "Max steps reached without completion"

# Example
result = react_agent(
    "Find the population of the capital city of France",
    functions=[SEARCH_FUNCTION, WIKIPEDIA_FUNCTION],
    function_registry={"search": search, "get_wikipedia": get_wikipedia}
)
\`\`\`

Output:
\`\`\`
--- Step 1 ---
THINK: I need to find the capital of France first.
ACT: search({"query": "capital of France"})
OBSERVE: {"result": "Paris"}

--- Step 2 ---
THINK: Now I know the capital is Paris. I need to find its population.
ACT: get_wikipedia({"article": "Paris"})
OBSERVE: {"population": "2.2 million", "metropolitan": "12.5 million"}

--- Step 3 ---
THINK: I have the population information.
FINAL ANSWER: The population of Paris, the capital city of France, is approximately 2.2 million in the city proper, and 12.5 million in the metropolitan area.
\`\`\`

**Use case**: Complex reasoning tasks, research, problem-solving.

## Error Recovery Patterns

Handle errors gracefully and retry with different strategies.

\`\`\`python
from typing import Optional
import time

def execute_with_retry (func_name: str, func_args: dict, 
                      function_registry: dict,
                      max_retries: int = 3,
                      backoff_factor: float = 2.0) -> dict:
    """
    Execute function with exponential backoff retry.
    """
    last_error = None
    
    for attempt in range (max_retries):
        try:
            result = function_registry[func_name](**func_args)
            return {
                "success": True,
                "data": result,
                "attempts": attempt + 1
            }
        except Exception as e:
            last_error = e
            wait_time = backoff_factor ** attempt
            print(f"Attempt {attempt + 1} failed: {e}")
            print(f"Retrying in {wait_time}s...")
            time.sleep (wait_time)
    
    return {
        "success": False,
        "error": str (last_error),
        "attempts": max_retries
    }

def workflow_with_fallback (user_message: str):
    """
    Workflow with fallback functions.
    """
    messages = [{"role": "user", "content": user_message}]
    
    # Try primary function
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        functions=[PRIMARY_SEARCH_FUNCTION]
    )
    
    message = response.choices[0].message
    messages.append (message.to_dict())
    
    if message.function_call:
        func_name = message.function_call.name
        func_args = json.loads (message.function_call.arguments)
        
        # Try primary function
        result = execute_with_retry (func_name, func_args, function_registry)
        
        if not result["success"]:
            # Primary failed, try fallback
            print("Primary function failed, trying fallback...")
            
            messages.append({
                "role": "function",
                "name": func_name,
                "content": json.dumps({
                    "error": "Primary search failed",
                    "fallback_available": True
                })
            })
            
            # Ask LLM to use fallback
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=messages,
                functions=[FALLBACK_SEARCH_FUNCTION]
            )
            
            message = response.choices[0].message
            messages.append (message.to_dict())
            
            if message.function_call:
                func_name = message.function_call.name
                func_args = json.loads (message.function_call.arguments)
                result = function_registry[func_name](**func_args)
                
                messages.append({
                    "role": "function",
                    "name": func_name,
                    "content": json.dumps (result)
                })
        else:
            messages.append({
                "role": "function",
                "name": func_name,
                "content": json.dumps (result["data"])
            })
    
    # Get final response
    final_response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    
    return final_response.choices[0].message.content
\`\`\`

## State Management

For complex workflows, maintain state explicitly:

\`\`\`python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

class WorkflowStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class WorkflowState:
    """State for a multi-step workflow."""
    user_goal: str
    status: WorkflowStatus
    steps: List[Dict[str, Any]]
    current_step: int
    data: Dict[str, Any]
    messages: List[Dict[str, Any]]
    
    def add_step (self, step_type: str, details: dict):
        """Add a completed step."""
        self.steps.append({
            "step_number": len (self.steps) + 1,
            "type": step_type,
            "details": details,
            "timestamp": time.time()
        })
        self.current_step = len (self.steps)
    
    def store_data (self, key: str, value: Any):
        """Store data from a step for use in later steps."""
        self.data[key] = value
    
    def get_data (self, key: str) -> Optional[Any]:
        """Retrieve stored data."""
        return self.data.get (key)

def stateful_workflow (user_goal: str, functions: list, function_registry: dict):
    """
    Workflow with explicit state management.
    """
    # Initialize state
    state = WorkflowState(
        user_goal=user_goal,
        status=WorkflowStatus.IN_PROGRESS,
        steps=[],
        current_step=0,
        data={},
        messages=[{"role": "user", "content": user_goal}]
    )
    
    max_steps = 10
    
    for step_num in range (max_steps):
        # Call LLM
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=state.messages,
            functions=functions
        )
        
        message = response.choices[0].message
        state.messages.append (message.to_dict())
        
        if message.function_call:
            func_name = message.function_call.name
            func_args = json.loads (message.function_call.arguments)
            
            # Execute function
            try:
                result = function_registry[func_name](**func_args)
                
                # Record step
                state.add_step("function_call", {
                    "function": func_name,
                    "arguments": func_args,
                    "result": result
                })
                
                # Store important data for later use
                if func_name == "search":
                    state.store_data("search_results", result)
                elif func_name == "get_user":
                    state.store_data("user_id", result.get("id"))
                
                # Add result to messages
                state.messages.append({
                    "role": "function",
                    "name": func_name,
                    "content": json.dumps (result)
                })
                
            except Exception as e:
                state.add_step("error", {
                    "function": func_name,
                    "error": str (e)
                })
                state.status = WorkflowStatus.FAILED
                return state
        else:
            # Final answer
            state.add_step("final_answer", {
                "content": message.content
            })
            state.status = WorkflowStatus.COMPLETED
            return state
    
    state.status = WorkflowStatus.FAILED
    return state

# Use it
state = stateful_workflow(
    "Book a restaurant and send confirmation email",
    functions=ALL_FUNCTIONS,
    function_registry=FUNCTION_REGISTRY
)

print(f"Status: {state.status}")
print(f"Steps completed: {state.current_step}")
for step in state.steps:
    print(f"  {step['step_number']}. {step['type']}: {step['details']}")
\`\`\`

## Workflow Orchestration Frameworks

For production systems, consider using frameworks:

### LangChain Agent Executor

\`\`\`python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

# Define tools
tools = [
    Tool(
        name="get_weather",
        func=get_weather,
        description="Get current weather for a location"
    ),
    Tool(
        name="search",
        func=search,
        description="Search the web"
    )
]

# Create agent
llm = ChatOpenAI(model="gpt-4", temperature=0)
agent = create_openai_functions_agent (llm, tools, prompt)

# Create executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    handle_parsing_errors=True
)

# Run
result = agent_executor.invoke({"input": "What\'s the weather in SF?"})
\`\`\`

### Custom Orchestrator

\`\`\`python
class WorkflowOrchestrator:
    """
    Production-grade workflow orchestrator.
    """
    def __init__(self, functions: list, function_registry: dict, 
                 max_iterations: int = 10):
        self.functions = functions
        self.function_registry = function_registry
        self.max_iterations = max_iterations
    
    async def execute_workflow (self, user_goal: str) -> WorkflowState:
        """Execute a workflow asynchronously."""
        state = WorkflowState(
            user_goal=user_goal,
            status=WorkflowStatus.IN_PROGRESS,
            steps=[],
            current_step=0,
            data={},
            messages=[{"role": "user", "content": user_goal}]
        )
        
        for iteration in range (self.max_iterations):
            # Get next action from LLM
            action = await self._get_next_action (state)
            
            if action["type"] == "function_call":
                # Execute function
                result = await self._execute_function(
                    action["function"],
                    action["arguments"]
                )
                state.add_step("function_call", {
                    "function": action["function"],
                    "result": result
                })
                
            elif action["type"] == "final_answer":
                state.status = WorkflowStatus.COMPLETED
                return state
            
            elif action["type"] == "error":
                state.status = WorkflowStatus.FAILED
                return state
        
        state.status = WorkflowStatus.FAILED
        return state
    
    async def _get_next_action (self, state: WorkflowState) -> dict:
        """Get next action from LLM."""
        # Implementation
        pass
    
    async def _execute_function (self, func_name: str, args: dict) -> Any:
        """Execute a function asynchronously."""
        # Implementation
        pass
\`\`\`

## Best Practices

1. **Set max iterations** to prevent infinite loops
2. **Handle errors gracefully** with try-except and fallbacks
3. **Log each step** for debugging
4. **Maintain conversation context** in messages array
5. **Use state management** for complex workflows
6. **Implement timeouts** for long-running functions
7. **Provide progress updates** for long workflows
8. **Allow cancellation** for user interruption
9. **Cache function results** when appropriate
10. **Monitor costs** - each LLM call adds up

## Summary

Workflow patterns:
- **Single call**: Simple queries
- **Sequential**: Multi-step dependent tasks
- **Parallel**: Independent tasks executed simultaneously
- **Conditional**: Branching logic based on results
- **Iterative**: Loops until condition met
- **ReAct**: Explicit reasoning + acting
- **Error recovery**: Retries and fallbacks

Choose the pattern based on your use case. Production systems often combine multiple patterns with proper state management and error handling.

Next, we'll explore building reusable tool libraries that can be composed into these workflows.
`,
};
