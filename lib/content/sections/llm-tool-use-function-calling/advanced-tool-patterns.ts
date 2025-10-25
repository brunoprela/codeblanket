export const advancedToolPatterns = {
  title: 'Advanced Tool Patterns',
  id: 'advanced-tool-patterns',
  description:
    'Explore advanced patterns like tool chaining, composition, dynamic tool generation, and conditional tool execution.',
  content: `

# Advanced Tool Patterns

## Introduction

Beyond basic tool use, advanced patterns enable sophisticated behaviors:
- **Tool Chaining**: Using output of one tool as input to another
- **Tool Composition**: Combining tools to create higher-level capabilities
- **Dynamic Tool Generation**: Creating tools on-the-fly based on context
- **Conditional Tools**: Tools that adapt based on state
- **Tool Factories**: Generating families of related tools
- **Meta-Tools**: Tools that create or modify other tools

These patterns unlock powerful capabilities for complex agentic systems.

## Tool Chaining

### Sequential Chaining

Automatically chain tools based on dependencies:

\`\`\`python
from typing import List, Dict, Any, Callable
from dataclasses import dataclass

@dataclass
class ToolChainStep:
    """Single step in a tool chain."""
    tool_name: str
    argument_mapping: Dict[str, str]  # Maps previous results to arguments
    condition: Callable[[Dict], bool] = None  # Optional condition

class ToolChain:
    """Chain multiple tools together."""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
    
    def execute_chain (self, 
                     steps: List[ToolChainStep], 
                     initial_args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute a chain of tools.
        
        Example chain:
        1. search_location (query) → location_data
        2. get_weather (location=location_data.name) → weather
        3. send_notification (message=weather.summary)
        """
        results = []
        context = initial_args.copy()
        
        for step in steps:
            # Check condition if present
            if step.condition and not step.condition (context):
                continue
            
            # Build arguments from context
            args = {}
            for param, mapping in step.argument_mapping.items():
                # Support nested access: "results[0].temperature"
                value = self._resolve_mapping (mapping, context)
                args[param] = value
            
            # Execute tool
            result = self.registry.execute (step.tool_name, **args)
            results.append (result)
            
            # Update context
            context[f"{step.tool_name}_result"] = result
        
        return results
    
    def _resolve_mapping (self, mapping: str, context: Dict) -> Any:
        """Resolve a mapping expression like 'search_result.location.name'."""
        parts = mapping.split('.')
        value = context
        
        for part in parts:
            if isinstance (value, dict):
                value = value.get (part)
            elif hasattr (value, part):
                value = getattr (value, part)
            else:
                return None
        
        return value

# Example: Weather notification chain
chain = ToolChain (registry)

steps = [
    ToolChainStep(
        tool_name="get_weather",
        argument_mapping={
            "location": "location"  # From initial args
        }
    ),
    ToolChainStep(
        tool_name="send_notification",
        argument_mapping={
            "message": "get_weather_result.data.summary"
        },
        condition=lambda ctx: ctx.get("get_weather_result", {}).get("status") == "success"
    )
]

results = chain.execute_chain (steps, {"location": "San Francisco"})
\`\`\`

### LLM-Guided Chaining

Let the LLM decide the chain:

\`\`\`python
def llm_guided_chain (user_goal: str, 
                    available_tools: List[Tool],
                    max_steps: int = 5) -> List[Dict[str, Any]]:
    """
    Let LLM decide tool chain to accomplish goal.
    """
    messages = [
        {
            "role": "system",
            "content": """You are planning a sequence of tool calls to accomplish a goal.
            
Think through the steps needed:
1. What tools do you need?
2. In what order?
3. How do results from one tool feed into the next?

Call tools one at a time, building on previous results."""
        },
        {
            "role": "user",
            "content": f"Goal: {user_goal}"
        }
    ]
    
    results = []
    context = {}
    
    for step in range (max_steps):
        # Get next action from LLM
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            functions=[tool.to_function_schema() for tool in available_tools]
        )
        
        message = response.choices[0].message
        messages.append (message.to_dict())
        
        if message.function_call:
            # Execute tool
            func_name = message.function_call.name
            func_args = json.loads (message.function_call.arguments)
            
            result = registry.execute (func_name, **func_args)
            results.append({
                "tool": func_name,
                "arguments": func_args,
                "result": result
            })
            
            # Add to messages
            messages.append({
                "role": "function",
                "name": func_name,
                "content": json.dumps (result)
            })
        else:
            # LLM finished the chain
            break
    
    return results

# Example
results = llm_guided_chain(
    "Find the weather in the capital of France and send it to my Slack",
    available_tools=[
        search_tool,
        weather_tool,
        slack_tool
    ]
)
\`\`\`

## Tool Composition

### Composite Tools

Create higher-level tools from primitives:

\`\`\`python
@tool (description="Research a topic comprehensively")
def research_topic (topic: str) -> dict:
    """
    Composite tool that uses multiple sources.
    
    Steps:
    1. Search web
    2. Search Wikipedia
    3. Search academic sources
    4. Synthesize results
    """
    # Step 1: Web search
    web_results = google_search (query=topic, num_results=5)
    
    # Step 2: Wikipedia
    wiki_results = wikipedia_search (query=topic)
    
    # Step 3: Academic search
    academic_results = google_scholar_search (query=topic, num_results=5)
    
    # Step 4: Synthesize with LLM
    synthesis_prompt = f"""Synthesize the following research on '{topic}':

Web results: {json.dumps (web_results)}
Wikipedia: {json.dumps (wiki_results)}
Academic: {json.dumps (academic_results)}

Provide a comprehensive summary covering:
1. Key concepts
2. Current state
3. Notable developments
4. Main sources of information"""
    
    synthesis = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": synthesis_prompt}]
    )
    
    return {
        "status": "success",
        "data": {
            "topic": topic,
            "sources": {
                "web": web_results,
                "wikipedia": wiki_results,
                "academic": academic_results
            },
            "synthesis": synthesis.choices[0].message.content
        }
    }
\`\`\`

### Tool Decorators for Composition

\`\`\`python
from functools import wraps

def retry_on_failure (max_retries: int = 3):
    """Decorator to retry tool on failure."""
    def decorator (func):
        @wraps (func)
        def wrapper(*args, **kwargs):
            for attempt in range (max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)  # Exponential backoff
        return wrapper
    return decorator

def with_fallback (fallback_tool: str):
    """Decorator to use fallback tool on failure."""
    def decorator (func):
        @wraps (func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Try fallback
                return registry.execute (fallback_tool, **kwargs)
        return wrapper
    return decorator

def cached (ttl_seconds: int = 3600):
    """Decorator to cache tool results."""
    cache = {}
    
    def decorator (func):
        @wraps (func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = f"{func.__name__}:{args}:{sorted (kwargs.items())}"
            
            # Check cache
            if key in cache:
                cached_result, cached_time = cache[key]
                if time.time() - cached_time < ttl_seconds:
                    return cached_result
            
            # Execute and cache
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            
            return result
        return wrapper
    return decorator

# Usage
@tool (description="Get weather with retries and caching")
@retry_on_failure (max_retries=3)
@with_fallback("get_weather_alternative")
@cached (ttl_seconds=1800)
def get_weather_robust (location: str, unit: str = "celsius") -> dict:
    """Robust weather tool with multiple resilience patterns."""
    return weather_api.get_weather (location, unit)
\`\`\`

## Dynamic Tool Generation

### Context-Based Tool Generation

Generate tools based on context:

\`\`\`python
class DynamicToolGenerator:
    """Generate tools dynamically based on context."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
    
    def generate_api_tool (self, 
                         endpoint: str, 
                         method: str,
                         description: str,
                         parameters: Dict) -> Tool:
        """
        Generate a tool for an API endpoint.
        """
        def api_call(**kwargs):
            response = requests.request(
                method=method,
                url=f"{self.base_url}/{endpoint}",
                json=kwargs,
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            return response.json()
        
        return Tool(
            name=f"api_{endpoint.replace('/', '_')}",
            description=description,
            function=api_call,
            schema=parameters,
            category=ToolCategory.API_INTEGRATION
        )
    
    def generate_tools_from_openapi (self, openapi_spec: Dict) -> List[Tool]:
        """
        Generate tools from OpenAPI spec.
        """
        tools = []
        
        for path, methods in openapi_spec.get("paths", {}).items():
            for method, spec in methods.items():
                if method.upper() not in ["GET", "POST", "PUT", "DELETE"]:
                    continue
                
                tool = self.generate_api_tool(
                    endpoint=path,
                    method=method.upper(),
                    description=spec.get("summary", ""),
                    parameters=spec.get("parameters", {})
                )
                
                tools.append (tool)
        
        return tools

# Usage
generator = DynamicToolGenerator(
    base_url="https://api.example.com",
    api_key="sk-..."
)

# Generate tools from OpenAPI spec
openapi_spec = requests.get("https://api.example.com/openapi.json").json()
tools = generator.generate_tools_from_openapi (openapi_spec)

# Register all generated tools
for tool in tools:
    registry.register (tool)
\`\`\`

### Database-Driven Tools

Generate tools from database schema:

\`\`\`python
class DatabaseToolGenerator:
    """Generate tools for database operations."""
    
    def __init__(self, connection_string: str):
        self.engine = create_engine (connection_string)
        self.inspector = inspect (self.engine)
    
    def generate_query_tool (self, table_name: str) -> Tool:
        """Generate a query tool for a table."""
        # Get table schema
        columns = self.inspector.get_columns (table_name)
        
        # Build parameter schema
        parameters = {
            "type": "object",
            "properties": {}
        }
        
        for col in columns:
            parameters["properties"][col["name"]] = {
                "type": self._sql_type_to_json_type (col["type"]),
                "description": f"Filter by {col['name']}"
            }
        
        # Create query function
        def query_table(**filters):
            query = f"SELECT * FROM {table_name}"
            
            if filters:
                conditions = [f"{k} = :{k}" for k in filters.keys()]
                query += f" WHERE {' AND '.join (conditions)}"
            
            with self.engine.connect() as conn:
                result = conn.execute (text (query), filters)
                return [dict (row) for row in result]
        
        return Tool(
            name=f"query_{table_name}",
            description=f"Query the {table_name} table",
            function=query_table,
            schema=parameters,
            category=ToolCategory.DATABASE
        )
    
    def generate_all_table_tools (self) -> List[Tool]:
        """Generate tools for all tables."""
        tools = []
        
        for table_name in self.inspector.get_table_names():
            tool = self.generate_query_tool (table_name)
            tools.append (tool)
        
        return tools

# Usage
db_generator = DatabaseToolGenerator("postgresql://localhost/mydb")
table_tools = db_generator.generate_all_table_tools()

for tool in table_tools:
    registry.register (tool)
\`\`\`

## Tool Factories

### Parameterized Tool Generation

\`\`\`python
class ToolFactory:
    """Factory for generating related tools."""
    
    @staticmethod
    def create_http_tool (name: str, 
                        url: str, 
                        method: str = "GET",
                        headers: Dict = None) -> Tool:
        """Create an HTTP request tool."""
        def http_request(**params):
            response = requests.request(
                method=method,
                url=url,
                params=params if method == "GET" else None,
                json=params if method != "GET" else None,
                headers=headers or {}
            )
            return response.json()
        
        return Tool(
            name=name,
            description=f"{method} request to {url}",
            function=http_request,
            schema={
                "type": "object",
                "properties": {
                    "params": {
                        "type": "object",
                        "description": "Request parameters"
                    }
                }
            },
            category=ToolCategory.API_INTEGRATION
        )
    
    @staticmethod
    def create_file_tool (operation: str) -> Tool:
        """Create a file operation tool."""
        def file_op (path: str, content: str = None):
            if operation == "read":
                with open (path, 'r') as f:
                    return {"content": f.read()}
            elif operation == "write":
                with open (path, 'w') as f:
                    f.write (content)
                return {"success": True}
            elif operation == "delete":
                os.remove (path)
                return {"success": True}
        
        return Tool(
            name=f"file_{operation}",
            description=f"{operation.capitalize()} a file",
            function=file_op,
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path"]
            },
            category=ToolCategory.FILE_SYSTEM
        )

# Create tools from factory
weather_tool = ToolFactory.create_http_tool(
    name="get_weather",
    url="https://api.weather.com/v1/weather",
    headers={"API-Key": "..."}
)

read_tool = ToolFactory.create_file_tool("read")
write_tool = ToolFactory.create_file_tool("write")
\`\`\`

## Meta-Tools

### Tools That Create Tools

\`\`\`python
@tool (description="Create a new tool from a description")
def create_tool_from_description (name: str, description: str, 
                                 code: str) -> dict:
    """
    Meta-tool that creates new tools.
    
    Uses LLM to generate tool code from description.
    """
    # Generate tool implementation
    prompt = f"""Create a Python function that implements this tool:

Name: {name}
Description: {description}
Code template: {code}

Return valid Python function code."""
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    generated_code = response.choices[0].message.content
    
    # Execute code to create function
    namespace = {}
    exec (generated_code, namespace)
    func = namespace[name]
    
    # Create and register tool
    new_tool = Tool(
        name=name,
        description=description,
        function=func,
        schema=generate_schema_from_function (func),
        category=ToolCategory.COMPUTATION
    )
    
    registry.register (new_tool)
    
    return {
        "status": "success",
        "tool_name": name,
        "message": f"Tool '{name}' created and registered"
    }
\`\`\`

### Tools That Modify Tools

\`\`\`python
@tool (description="Modify an existing tool's behavior")
def add_logging_to_tool (tool_name: str) -> dict:
    """Add logging to an existing tool."""
    tool = registry.get (tool_name)
    
    if not tool:
        return {"status": "error", "error": f"Tool {tool_name} not found"}
    
    # Wrap function with logging
    original_func = tool.function
    
    def logged_func(**kwargs):
        logger.info (f"Calling {tool_name} with {kwargs}")
        result = original_func(**kwargs)
        logger.info (f"{tool_name} returned {result}")
        return result
    
    # Update tool
    tool.function = logged_func
    
    return {
        "status": "success",
        "message": f"Added logging to {tool_name}"
    }

@tool (description="Combine two tools into one")
def combine_tools (tool1_name: str, tool2_name: str, 
                 new_name: str) -> dict:
    """Create a new tool that combines two tools."""
    tool1 = registry.get (tool1_name)
    tool2 = registry.get (tool2_name)
    
    if not tool1 or not tool2:
        return {"status": "error", "error": "One or both tools not found"}
    
    def combined_func(**kwargs):
        result1 = tool1.execute(**kwargs)
        result2 = tool2.execute(**kwargs)
        return {
            f"{tool1_name}_result": result1,
            f"{tool2_name}_result": result2
        }
    
    new_tool = Tool(
        name=new_name,
        description=f"Combines {tool1_name} and {tool2_name}",
        function=combined_func,
        schema=tool1.schema,  # Simplified
        category=tool1.category
    )
    
    registry.register (new_tool)
    
    return {
        "status": "success",
        "tool_name": new_name
    }
\`\`\`

## Conditional Tools

### State-Based Tool Selection

\`\`\`python
class ConditionalToolRegistry(ToolRegistry):
    """Registry that provides tools based on state."""
    
    def __init__(self):
        super().__init__()
        self.state = {}
    
    def get_available_tools (self, context: Dict) -> List[Tool]:
        """Get tools available in current context."""
        available = []
        
        for tool in self.get_all():
            # Check if tool should be available
            if self._should_be_available (tool, context):
                available.append (tool)
        
        return available
    
    def _should_be_available (self, tool: Tool, context: Dict) -> bool:
        """Determine if tool should be available."""
        # Example: Admin tools only for admins
        if tool.category == ToolCategory.ADMIN:
            return context.get("user", {}).get("is_admin", False)
        
        # Example: Expensive tools only when budget available
        if tool.estimated_cost > 0.01:
            return context.get("budget_remaining", 0) > tool.estimated_cost
        
        return True

# Usage
conditional_registry = ConditionalToolRegistry()

context = {
    "user": {"id": "user_123", "is_admin": False},
    "budget_remaining": 0.05
}

available_tools = conditional_registry.get_available_tools (context)
\`\`\`

## Tool Pipeline Pattern

\`\`\`python
from typing import Callable, Any

class ToolPipeline:
    """Pipeline of tool transformations."""
    
    def __init__(self):
        self.steps: List[Callable] = []
    
    def add_step (self, func: Callable) -> 'ToolPipeline':
        """Add a step to the pipeline."""
        self.steps.append (func)
        return self  # For chaining
    
    def execute (self, initial_input: Any) -> Any:
        """Execute pipeline."""
        result = initial_input
        
        for step in self.steps:
            result = step (result)
        
        return result

# Example: Data processing pipeline
pipeline = (ToolPipeline()
    .add_step (lambda x: fetch_data (x))
    .add_step (lambda x: clean_data (x))
    .add_step (lambda x: transform_data (x))
    .add_step (lambda x: analyze_data (x))
    .add_step (lambda x: generate_report (x))
)

result = pipeline.execute({"query": "sales data"})
\`\`\`

## Summary

Advanced tool patterns enable:
1. **Tool Chaining** - Sequential tool execution
2. **Tool Composition** - Combining tools for complex behaviors
3. **Dynamic Generation** - Creating tools on-the-fly
4. **Tool Factories** - Systematic tool creation
5. **Meta-Tools** - Tools that create/modify tools
6. **Conditional Tools** - Context-aware tool availability
7. **Tool Pipelines** - Data transformation workflows

These patterns are essential for building sophisticated agentic systems.

Next, we'll put it all together to build a complete agentic system from scratch.
`,
};
