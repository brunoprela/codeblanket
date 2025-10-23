export const functionCallingFundamentals = {
  title: 'Function Calling Fundamentals',
  id: 'function-calling-fundamentals',
  description:
    'Understanding the basics of function calling in LLMs and how modern AI systems use tools to extend their capabilities beyond text generation.',
  content: `

# Function Calling Fundamentals

## Introduction

Function calling is one of the most powerful features in modern Large Language Models, enabling them to interact with external systems, retrieve real-time information, and perform actions beyond generating text. This capability transforms LLMs from passive text generators into active agents that can use tools to accomplish complex tasks.

In this section, we'll explore how function calling works, why it's revolutionary, and how to implement it in production systems. We'll see how ChatGPT uses it to browse the web, execute code, and generate images, and how you can build similar capabilities in your own applications.

## What is Function Calling?

Function calling (also called "tool use" in some systems like Claude) allows an LLM to:

1. **Recognize when it needs external information or capabilities**
2. **Choose appropriate functions to call** from a provided set of tools
3. **Generate properly formatted function arguments** based on the conversation context
4. **Process function results** and incorporate them into its response

Instead of the LLM trying to answer everything from its training data, it can delegate specific tasks to specialized functions that you provide.

### The Problem Function Calling Solves

Before function calling, if you asked an LLM "What's the weather in San Francisco?", it could only:
- Make up a plausible-sounding answer (hallucination)
- Admit it doesn't have real-time data
- Try to extract weather information from its training data (outdated)

With function calling, the LLM can:
1. Recognize it needs real-time weather data
2. Call a \`get_weather(location="San Francisco")\` function
3. Receive actual current weather data
4. Incorporate that data into a natural response

This transforms LLMs from static knowledge bases into dynamic assistants that can interact with the real world.

## How Function Calling Works

Here's the typical flow:

\`\`\`
User: "What's the weather in San Francisco and New York?"
  ↓
LLM: Analyzes request, determines it needs weather data
  ↓
LLM: Generates function call JSON
{
  "name": "get_weather",
  "arguments": {"location": "San Francisco"}
}
  ↓
Your Code: Executes the function, returns result
{"temperature": 72, "conditions": "sunny"}
  ↓
LLM: Receives result, generates another function call
{
  "name": "get_weather",
  "arguments": {"location": "New York"}
}
  ↓
Your Code: Executes second function
{"temperature": 65, "conditions": "cloudy"}
  ↓
LLM: Synthesizes results into natural language
"In San Francisco, it's currently 72°F and sunny. 
In New York, it's 65°F and cloudy."
\`\`\`

The key insight: **The LLM doesn't actually execute the functions**. It generates structured JSON representing the function call, your code executes it, and you feed the results back to the LLM.

## OpenAI Function Calling API

Let's see a concrete example using OpenAI's API:

\`\`\`python
import openai
import json

# Define the function schema
functions = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g., San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use"
                }
            },
            "required": ["location"]
        }
    }
]

# Make the API call
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "What's the weather like in San Francisco?"}
    ],
    functions=functions,
    function_call="auto"  # Let the model decide when to call functions
)

message = response.choices[0].message

# Check if the model wants to call a function
if message.function_call:
    function_name = message.function_call.name
    function_args = json.loads(message.function_call.arguments)
    
    print(f"Function to call: {function_name}")
    print(f"Arguments: {function_args}")
    
    # Output:
    # Function to call: get_weather
    # Arguments: {'location': 'San Francisco, CA', 'unit': 'fahrenheit'}
\`\`\`

The model doesn't just extract "San Francisco" - it intelligently formatted it as "San Francisco, CA" based on the description, and chose a reasonable default unit.

## Anthropic Claude Tool Use

Claude uses a slightly different terminology ("tools" instead of "functions") but the concept is identical:

\`\`\`python
import anthropic

client = anthropic.Anthropic()

# Define tools
tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g., San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit"
                }
            },
            "required": ["location"]
        }
    }
]

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    messages=[
        {"role": "user", "content": "What's the weather in San Francisco?"}
    ]
)

# Check for tool use
for block in response.content:
    if block.type == "tool_use":
        print(f"Tool: {block.name}")
        print(f"Input: {block.input}")
\`\`\`

## Complete Function Calling Flow

Let's build a complete example that handles the full cycle:

\`\`\`python
import openai
import json
from typing import Dict, Any

def get_weather(location: str, unit: str = "fahrenheit") -> Dict[str, Any]:
    """
    Simulated weather API call.
    In production, this would call a real weather API.
    """
    # Simulate API call
    weather_data = {
        "San Francisco, CA": {"temp": 72, "conditions": "sunny"},
        "New York, NY": {"temp": 65, "conditions": "cloudy"},
        "Seattle, WA": {"temp": 58, "conditions": "rainy"}
    }
    
    data = weather_data.get(location, {"temp": 70, "conditions": "unknown"})
    
    return {
        "location": location,
        "temperature": data["temp"],
        "unit": unit,
        "conditions": data["conditions"]
    }

# Function registry - maps function names to actual Python functions
FUNCTION_REGISTRY = {
    "get_weather": get_weather
}

# Function schemas for the LLM
FUNCTION_SCHEMAS = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g., San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit"
                }
            },
            "required": ["location"]
        }
    }
]

def chat_with_functions(user_message: str, max_iterations: int = 5):
    """
    Main chat loop that handles function calling.
    """
    messages = [{"role": "user", "content": user_message}]
    
    for iteration in range(max_iterations):
        # Call the LLM
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            functions=FUNCTION_SCHEMAS,
            function_call="auto"
        )
        
        message = response.choices[0].message
        messages.append(message.to_dict())
        
        # Check if the model wants to call a function
        if message.function_call:
            function_name = message.function_call.name
            function_args = json.loads(message.function_call.arguments)
            
            print(f"🔧 Calling function: {function_name}")
            print(f"📥 Arguments: {function_args}")
            
            # Execute the function
            if function_name in FUNCTION_REGISTRY:
                function_result = FUNCTION_REGISTRY[function_name](**function_args)
                print(f"📤 Result: {function_result}")
                
                # Add function result to messages
                messages.append({
                    "role": "function",
                    "name": function_name,
                    "content": json.dumps(function_result)
                })
            else:
                # Function not found
                messages.append({
                    "role": "function",
                    "name": function_name,
                    "content": json.dumps({"error": f"Function {function_name} not found"})
                })
        else:
            # No function call, we have a final response
            return message.content
    
    return "Max iterations reached"

# Test it out
response = chat_with_functions(
    "What's the weather in San Francisco and New York? Which city is warmer?"
)
print(f"\\n🤖 Final response:\\n{response}")
\`\`\`

Output:
\`\`\`
🔧 Calling function: get_weather
📥 Arguments: {'location': 'San Francisco, CA', 'unit': 'fahrenheit'}
📤 Result: {'location': 'San Francisco, CA', 'temperature': 72, 'unit': 'fahrenheit', 'conditions': 'sunny'}

🔧 Calling function: get_weather
📥 Arguments: {'location': 'New York, NY', 'unit': 'fahrenheit'}
📤 Result: {'location': 'New York, NY', 'temperature': 65, 'unit': 'fahrenheit', 'conditions': 'cloudy'}

🤖 Final response:
In San Francisco, it's currently 72°F and sunny. In New York, it's 65°F and cloudy. San Francisco is warmer by 7 degrees.
\`\`\`

## When to Use Function Calling

Function calling is beneficial when you need to:

### 1. **Access Real-Time Data**
- Current weather, stock prices, news
- Database queries
- API responses
- Live metrics and analytics

### 2. **Perform Actions**
- Send emails or messages
- Create calendar events
- Update databases
- File operations
- API mutations

### 3. **Retrieve Specific Information**
- User account details
- Document contents
- Search results
- Product catalogs

### 4. **Execute Code**
- Run Python scripts
- Perform calculations
- Data analysis
- Code validation

### 5. **Structured Output**
- Extract data in specific formats
- Generate JSON/XML
- Form filling
- Data transformation

## Benefits Over Text Parsing

Before function calling, you might try to extract structured information from the LLM's text output:

\`\`\`python
# Old approach - unreliable
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{
        "role": "user",
        "content": "Extract the city from: 'What's the weather in San Francisco?'"
    }]
)

# Try to parse the city from text
city = response.choices[0].message.content.strip()
# Hope it returns just "San Francisco" and not "The city is San Francisco" or something else
\`\`\`

Problems with this approach:
- Unreliable parsing
- Need complex regex/extraction logic
- Format inconsistencies
- Error-prone
- Hard to validate

With function calling:

\`\`\`python
# New approach - reliable
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{
        "role": "user",
        "content": "What's the weather in San Francisco?"
    }],
    functions=[{
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }]
)

# Get structured function call
if response.choices[0].message.function_call:
    args = json.loads(response.choices[0].message.function_call.arguments)
    city = args["location"]  # Guaranteed to be a string
\`\`\`

## Common Use Cases

### ChatGPT Examples

**1. Web Browsing**
- Function: \`bing_search(query)\`
- Allows ChatGPT to search the web for current information
- Returns search results that ChatGPT can read and synthesize

**2. Code Interpreter**
- Function: \`execute_python(code)\`
- Runs Python code in a sandboxed environment
- Returns execution results, plots, and files

**3. DALL-E Image Generation**
- Function: \`generate_image(prompt, size, quality)\`
- Generates images based on text descriptions
- Returns image URLs

**4. Plugins**
- Various third-party functions
- Restaurant reservations, travel booking, shopping, etc.

### Cursor Examples

**1. File Reading**
- Function: \`read_file(path)\`
- Reads file contents for context
- Returns file text

**2. Code Search**
- Function: \`search_codebase(query)\`
- Semantic search across project
- Returns relevant code snippets

**3. Terminal Execution**
- Function: \`run_command(command)\`
- Executes terminal commands
- Returns stdout/stderr

## Function Call Format

The LLM generates function calls in a structured JSON format:

\`\`\`json
{
  "name": "function_name",
  "arguments": {
    "param1": "value1",
    "param2": 42,
    "param3": ["array", "values"],
    "param4": {
      "nested": "object"
    }
  }
}
\`\`\`

Key points:
- **name**: Exact function name (case-sensitive)
- **arguments**: JSON object with parameter values
- **Types preserved**: Strings, numbers, booleans, arrays, objects
- **Schema compliance**: Arguments should match the schema you provided

## Multiple Function Calls

Modern LLMs can make multiple function calls in parallel:

\`\`\`python
response = openai.chat.completions.create(
    model="gpt-4-1106-preview",  # Supports parallel function calling
    messages=[{
        "role": "user",
        "content": "What's the weather in SF, NYC, and LA?"
    }],
    functions=FUNCTION_SCHEMAS,
    function_call="auto"
)

# May return multiple function calls
for choice in response.choices:
    if choice.message.function_call:
        # Handle each function call
        pass
\`\`\`

This is more efficient than making sequential calls.

## Function Call Modes

OpenAI provides three modes for the \`function_call\` parameter:

### 1. \`auto\` (Default)
Let the model decide whether to call a function:
\`\`\`python
function_call="auto"
\`\`\`
- Model chooses based on context
- Can respond without calling any function
- Best for general assistants

### 2. \`none\`
Force the model to NOT call any functions:
\`\`\`python
function_call="none"
\`\`\`
- Model will respond with text only
- Useful when you temporarily disable tools
- Good for A/B testing

### 3. Specific Function
Force the model to call a specific function:
\`\`\`python
function_call={"name": "get_weather"}
\`\`\`
- Model MUST call this function
- Guaranteed function call
- Useful for structured data extraction

## Error Handling

Function calling can fail in various ways:

\`\`\`python
def safe_function_call(function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a function with comprehensive error handling.
    """
    try:
        # Check if function exists
        if function_name not in FUNCTION_REGISTRY:
            return {
                "error": f"Function '{function_name}' not found",
                "available_functions": list(FUNCTION_REGISTRY.keys())
            }
        
        # Get the function
        func = FUNCTION_REGISTRY[function_name]
        
        # Validate arguments (basic check)
        import inspect
        sig = inspect.signature(func)
        
        # Execute function
        result = func(**arguments)
        
        return {
            "success": True,
            "data": result
        }
        
    except TypeError as e:
        # Wrong arguments
        return {
            "error": f"Invalid arguments: {str(e)}",
            "function": function_name,
            "provided_args": arguments
        }
    
    except Exception as e:
        # Any other error
        return {
            "error": f"Function execution failed: {str(e)}",
            "function": function_name,
            "type": type(e).__name__
        }
\`\`\`

## Best Practices

### 1. Clear Function Descriptions
The LLM relies on descriptions to choose functions:

❌ Bad:
\`\`\`python
{
    "name": "get_data",
    "description": "Gets data"
}
\`\`\`

✅ Good:
\`\`\`python
{
    "name": "get_current_weather",
    "description": "Get the current weather conditions for a specific location. Returns temperature, conditions, humidity, and wind speed. Use this when users ask about current weather, not for forecasts."
}
\`\`\`

### 2. Descriptive Parameter Names
\`\`\`python
"parameters": {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "The city and state (e.g., 'San Francisco, CA') or ZIP code"
        },
        "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "Temperature unit. Use 'celsius' for metric, 'fahrenheit' for imperial."
        }
    }
}
\`\`\`

### 3. Use Enums for Constrained Values
\`\`\`python
"unit": {
    "type": "string",
    "enum": ["celsius", "fahrenheit"]
}
\`\`\`

This prevents the LLM from generating invalid values.

### 4. Mark Required Parameters
\`\`\`python
"required": ["location"]
\`\`\`

This ensures the LLM always provides essential parameters.

### 5. Provide Examples in Descriptions
\`\`\`python
"date": {
    "type": "string",
    "description": "Date in ISO 8601 format, e.g., '2024-01-15' or '2024-01-15T10:30:00Z'"
}
\`\`\`

## Production Considerations

### Rate Limiting
Function calls count as additional API calls:
- Initial call to generate function call
- Follow-up call with function result
- Potentially multiple iterations

Monitor your usage and costs.

### Timeout Handling
Functions might take time to execute:

\`\`\`python
import asyncio
from functools import wraps
import signal

def timeout(seconds):
    """Decorator to add timeout to functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set timeout
            def handler(signum, frame):
                raise TimeoutError(f"Function timed out after {seconds} seconds")
            
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)  # Cancel alarm
            
            return result
        return wrapper
    return decorator

@timeout(5)
def slow_api_call():
    # Function that might be slow
    pass
\`\`\`

### Security
Never execute arbitrary code from function arguments:

❌ Dangerous:
\`\`\`python
def execute_python(code: str):
    exec(code)  # NEVER DO THIS
\`\`\`

✅ Safe:
\`\`\`python
def execute_python(code: str):
    # Use sandbox, validate code, restrict imports, etc.
    result = safe_sandbox_executor.run(code, timeout=5, memory_limit=128)
    return result
\`\`\`

## Summary

Function calling is a transformative feature that enables LLMs to:
- Access real-time data
- Perform actions in external systems
- Retrieve specific information
- Execute code safely
- Generate structured outputs reliably

Key takeaways:
1. LLMs generate function call JSON; your code executes it
2. Clear descriptions and schemas are crucial
3. Handle the full cycle: request → function call → execution → result → response
4. Multiple iterations may be needed for complex tasks
5. Production systems need robust error handling and security

In the next section, we'll dive deeper into defining functions and tools with proper schemas, type systems, and documentation.

## Code Examples Repository

All complete code examples from this section:
- Basic function calling setup
- Complete chat loop with function calling
- Error handling patterns
- Production-ready function registry

Practice building a simple function-calling assistant before moving to more complex patterns.
`,
};
