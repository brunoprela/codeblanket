export const toolUsePrompting = {
  title: 'Tool Use Prompting',
  id: 'tool-use-prompting',
  description:
    'Master prompt engineering techniques specifically for teaching LLMs when and how to use tools effectively.',
  content: `

# Tool Use Prompting

## Introduction

Even with perfectly defined tools, an LLM won't use them effectively without proper prompting. Tool use prompting is a specialized form of prompt engineering that teaches the LLM:

1. **When** to use tools vs. answer directly
2. **Which** tool to use for a given task
3. **How** to formulate function arguments correctly
4. **What to do** with function results
5. **How to handle** errors and edge cases

In this section, we'll master the art of prompting for effective tool use, exploring system prompts, examples, constraints, and debugging techniques.

## System Prompt Fundamentals

The system prompt sets expectations for tool use:

\`\`\`python
BASE_SYSTEM_PROMPT = """You are a helpful AI assistant with access to various tools.

When the user asks a question:
1. Determine if you can answer directly from your knowledge
2. If you need current information or need to perform an action, use the appropriate tool
3. After receiving tool results, synthesize them into a clear, natural response

Available tools and when to use them:
- get_weather: For current weather conditions
- search_web: For recent information not in your training data
- query_database: For data from the company database
- send_email: To send emails (requires user confirmation)
- execute_python: For calculations and data analysis

Always choose the most appropriate tool for the task. If multiple tools could work, prefer the most specific one.
"""

response = openai.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": "What\'s the weather in Tokyo?"}
    ],
    functions=[weather_tool, search_tool, database_tool]
)
\`\`\`

## Teaching When to Use Tools

### Explicit Guidelines

Be explicit about when tools are needed vs. when knowledge is sufficient:

\`\`\`python
WHEN_TO_USE_TOOLS = """Tool Usage Guidelines:

USE TOOLS when:
- User asks about current/real-time information (weather, stock prices, news)
- User requests an action (send email, create calendar event, update database)
- User needs data you don't have (database queries, file contents)
- User needs calculations beyond simple math (data analysis, complex formulas)
- User asks about information after your training cutoff date

DO NOT use tools when:
- You can answer from your training knowledge
- Question is about general facts, concepts, or explanations
- Simple calculations you can do mentally
- User asks about your capabilities or how you work

When in doubt, use a tool to provide the most accurate and current information.
"""

system_prompt = BASE_SYSTEM_PROMPT + "\\n\\n" + WHEN_TO_USE_TOOLS
\`\`\`

### Examples in System Prompt

Provide clear examples:

\`\`\`python
TOOL_USE_EXAMPLES = """Examples:

User: "What's the capital of France?"
Response: No tool needed. Answer: "The capital of France is Paris."

User: "What\'s the weather in Paris?"
Tool: get_weather (location="Paris, France")
Response: Based on weather data...

User: "How many users signed up last week?"
Tool: query_database (query="SELECT COUNT(*) FROM users WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'")
Response: Based on the database...

User: "Send an email to john@example.com about the meeting"
Tool: send_email (to="john@example.com", subject="Meeting", body="...")
Response: I've sent the email to john@example.com.
"""

system_prompt = BASE_SYSTEM_PROMPT + "\\n\\n" + WHEN_TO_USE_TOOLS + "\\n\\n" + TOOL_USE_EXAMPLES
\`\`\`

## Teaching Which Tool to Choose

### Tool Descriptions Matter

Your tool descriptions are critical:

\`\`\`python
# ❌ BAD: Vague descriptions
{
    "name": "search",
    "description": "Search for things"
}

# ✅ GOOD: Clear, specific descriptions
{
    "name": "search_web",
    "description": """Search the internet for current information.
    
    Use when:
    - User asks about recent events or news
    - User needs information published after my training data
    - User wants to find websites or resources
    
    Do NOT use when:
    - User asks about general knowledge I should know
    - User asks about the company database (use query_database instead)
    - User wants weather (use get_weather instead)
    """
}
\`\`\`

### Tool Selection Prompting

Help the LLM choose between similar tools:

\`\`\`python
TOOL_SELECTION_GUIDANCE = """Tool Selection Guide:

For SEARCH tasks:
- search_web: General internet search
- search_wikipedia: Encyclopedic knowledge
- search_products: Company product catalog
- search_docs: Internal documentation

For DATA tasks:
- query_database: Structured data in database
- read_file: Read contents of a specific file
- execute_python: Analyze or transform data

For COMMUNICATION:
- send_email: Professional, documented communication
- send_slack: Quick team updates
- send_sms: Urgent notifications

Choose the MOST SPECIFIC tool that matches the need.
"""
\`\`\`

## Teaching Correct Argument Formatting

### Parameter Examples

Include examples in parameter descriptions:

\`\`\`python
{
    "name": "get_weather",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": """Location to get weather for.
                
                Supported formats:
                - City name: "London" or "London, UK"
                - US city: "Seattle, WA" or "Seattle, Washington"
                - Coordinates: "51.5074,-0.1278" (latitude,longitude)
                - ZIP code: "90210" (US only)
                
                Examples:
                - "San Francisco, CA"
                - "Tokyo, Japan"
                - "90210"
                - "51.5074,-0.1278"
                """
            },
            "date": {
                "type": "string",
                "description": """Date for weather forecast (optional, defaults to today).
                
                Format: YYYY-MM-DD
                Examples: "2024-01-15", "2024-12-25"
                """
            }
        }
    }
}
\`\`\`

### Argument Validation Prompting

Teach the LLM to validate before calling:

\`\`\`python
ARGUMENT_VALIDATION_PROMPT = """Before calling any tool:

1. Check you have all required parameters
2. Verify parameter formats match the schema
3. If user's request is ambiguous, make reasonable assumptions
4. If critical information is missing, ask the user for clarification

Examples:

User: "What\'s the weather?"
Missing: location
Action: Ask user "Which location would you like weather for?"

User: "What's the weather in SF?"
Ambiguous: "SF" could be San Francisco or Santa Fe
Action: Assume "San Francisco, CA" (most common)

User: "Send an email"
Missing: recipient, subject, body
Action: Ask user "Who should I send the email to, and what should it say?"
"""
\`\`\`

## Teaching Result Processing

### Synthesizing Results

Teach how to use tool results:

\`\`\`python
RESULT_PROCESSING_PROMPT = """After receiving tool results:

1. Analyze the result for relevant information
2. Extract key details the user cares about
3. Synthesize into natural language
4. If result contains an error, explain it clearly
5. If more information is needed, call another tool

Examples:

Tool result: {"temperature": 72, "conditions": "sunny", "humidity": 45}
Response: "It's currently 72°F and sunny in San Francisco, with humidity at 45%."

Tool result: {"error": "Location not found"}
Response: "I couldn't find weather data for that location. Could you provide a more specific location name or try a nearby city?"

Tool result: {"count": 42}
Response: "There are 42 users who signed up last week."
"""
\`\`\`

### Multi-Step Reasoning

For complex queries requiring multiple tools:

\`\`\`python
MULTI_STEP_PROMPT = """For complex requests:

1. Break down the request into steps
2. Execute tools in logical order
3. Use results from one tool to inform the next
4. Synthesize all results into a comprehensive answer

Example:

User: "What\'s warmer, San Francisco or Seattle right now?"

Step 1: Get weather for San Francisco
Tool: get_weather (location="San Francisco, CA")
Result: 68°F

Step 2: Get weather for Seattle
Tool: get_weather (location="Seattle, WA")
Result: 55°F

Step 3: Compare and respond
Response: "San Francisco is warmer at 68°F, compared to Seattle's 55°F - a difference of 13 degrees."
"""
\`\`\`

## Error Handling Prompts

### Teaching Error Recovery

\`\`\`python
ERROR_HANDLING_PROMPT = """If a tool call fails:

1. Check the error message
2. Determine if you can retry with different arguments
3. If retry is appropriate, try a modified approach
4. If not recoverable, explain the issue to the user

Examples:

Error: "Rate limit exceeded"
Action: Wait a moment and inform user: "The API is temporarily rate limited. Let me try again in a moment."

Error: "Invalid location"
Action: Ask user: "I couldn't find that location. Could you provide the full city name and country/state?"

Error: "Permission denied"
Action: Explain: "I don't have permission to access that resource. You may need to grant me access or ask your administrator."

Error: "Timeout"
Action: Retry once, then: "The request timed out. The service might be slow or unavailable. Would you like me to try again?"
"""
\`\`\`

## Constraint Prompting

### Setting Boundaries

\`\`\`python
TOOL_CONSTRAINTS_PROMPT = """Tool Usage Constraints:

ALWAYS:
- Confirm before sending emails or making changes
- Explain what you're about to do with a tool
- Validate tool results before using them
- Handle errors gracefully

NEVER:
- Make multiple identical tool calls
- Use tools for information you already have
- Call tools with obviously invalid parameters
- Ignore errors from tools

Rate Limits:
- Maximum 5 tool calls per response
- If you need more, explain why and ask permission

Cost Awareness:
- search_web: $0.001 per query
- execute_python: $0.002 per execution
- query_database: Free
- Be mindful of costs for expensive operations
"""
\`\`\`

## Few-Shot Examples for Tool Use

Provide complete examples in conversation format:

\`\`\`python
FEW_SHOT_EXAMPLES = """Here are examples of correct tool usage:

Example 1: Simple query
User: "What\'s 15% of 230?"
Assistant: Let me calculate that for you.
Tool: execute_python (code="result = 0.15 * 230\\nprint(result)")
Result: 34.5
Assistant: 15% of 230 is 34.5.

Example 2: Multi-step query
User: "Search for Python tutorials and summarize the top result"
Assistant: I'll search for Python tutorials.
Tool: search_web (query="Python tutorials")
Result: [{"title": "Learn Python", "url": "...", "snippet": "..."}]
Assistant: I found several tutorials. Let me get more details on the top one.
Tool: read_url (url="...")
Result: {"content": "..."}
Assistant: Based on the top result, here's a summary: [summary]

Example 3: Error handling
User: "What\'s the weather in Atlantis?"
Assistant: Let me check the weather in Atlantis.
Tool: get_weather (location="Atlantis")
Result: {"error": "Location not found"}
Assistant: I couldn't find a location called "Atlantis". Did you mean Atlanta, or perhaps a different city?
"""

messages = [
    {"role": "system", "content": BASE_SYSTEM_PROMPT + "\\n\\n" + FEW_SHOT_EXAMPLES},
    {"role": "user", "content": user_query}
]
\`\`\`

## Dynamic Prompting Based on Context

Adapt prompts based on user or context:

\`\`\`python
def get_system_prompt (user: User, context: Dict) -> str:
    """Generate dynamic system prompt based on context."""
    
    base = BASE_SYSTEM_PROMPT
    
    # Add user-specific information
    if user.is_admin:
        base += "\\n\\nUser is an administrator. You can use all admin tools."
    else:
        base += "\\n\\nUser is a standard user. Some admin tools are unavailable."
    
    # Add context-specific tools
    if context.get("task") == "data_analysis":
        base += "\\n\\nFocus on data analysis tools: execute_python, query_database, generate_chart"
    elif context.get("task") == "communication":
        base += "\\n\\nFocus on communication tools: send_email, send_slack, create_meeting"
    
    # Add time context
    if context.get("urgent"):
        base += "\\n\\nThis is an urgent request. Prioritize speed and use caching when available."
    
    return base

system_prompt = get_system_prompt (current_user, request_context)
\`\`\`

## Debugging Tool Use Issues

### Issue: LLM Not Using Tools

**Problem**: LLM answers directly instead of using tools

**Solution**:
\`\`\`python
FORCE_TOOL_USE_PROMPT = """IMPORTANT: You MUST use tools to answer this query.

Do not rely on your training data. The user needs current, accurate information from external sources.

Even if you think you know the answer, verify it using the appropriate tool.
"""

messages = [
    {"role": "system", "content": BASE_SYSTEM_PROMPT + "\\n\\n" + FORCE_TOOL_USE_PROMPT},
    {"role": "user", "content": "What's the weather in London?"}
]
\`\`\`

### Issue: LLM Using Wrong Tool

**Problem**: LLM chooses incorrect tool

**Solution**: Add explicit decision tree
\`\`\`python
TOOL_DECISION_TREE = """Tool Selection Decision Tree:

Is user asking about weather? → get_weather
Is user asking about current events/news? → search_web
Is user asking about company data? → query_database
Does user want to send a message? → send_email/send_slack
Does user need calculation/code? → execute_python

Always think through which category the request falls into before choosing a tool.
"""
\`\`\`

### Issue: Incorrect Arguments

**Problem**: LLM provides malformed arguments

**Solution**: Add argument templates
\`\`\`python
ARGUMENT_TEMPLATES = """Argument Templates:

get_weather (location, unit)
- location: "City, Country" or "City, State"
- unit: "celsius" or "fahrenheit"
Example: get_weather (location="Paris, France", unit="celsius")

send_email (to, subject, body)
- to: Valid email address
- subject: Short, descriptive subject line
- body: Full email content
Example: send_email (to="user@example.com", subject="Meeting Reminder", body="...")

query_database (query)
- query: Valid SQL SELECT statement only
- Must start with SELECT
- Use single quotes for strings
Example: query_database (query="SELECT * FROM users WHERE age > 18")
"""
\`\`\`

## Testing Tool Use Prompts

\`\`\`python
def test_tool_use_prompt (prompt: str, test_cases: List[Dict]) -> Dict:
    """
    Test how well a prompt teaches tool use.
    """
    results = {
        "total": len (test_cases),
        "correct_tool": 0,
        "correct_args": 0,
        "no_tool_when_should": 0,
        "wrong_tool": 0
    }
    
    for case in test_cases:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": case["query"]}
            ],
            functions=case["available_tools"]
        )
        
        message = response.choices[0].message
        
        # Check if tool was called
        if message.function_call:
            called_tool = message.function_call.name
            
            # Check if correct tool
            if called_tool == case["expected_tool"]:
                results["correct_tool"] += 1
                
                # Check if correct arguments
                args = json.loads (message.function_call.arguments)
                if args == case["expected_args"]:
                    results["correct_args"] += 1
            else:
                results["wrong_tool"] += 1
        else:
            if case["expected_tool"]:
                results["no_tool_when_should"] += 1
    
    # Calculate scores
    results["tool_selection_rate"] = results["correct_tool"] / results["total"]
    results["argument_accuracy_rate"] = results["correct_args"] / results["total"]
    
    return results

# Test cases
test_cases = [
    {
        "query": "What\'s the weather in Tokyo?",
        "expected_tool": "get_weather",
        "expected_args": {"location": "Tokyo, Japan"},
        "available_tools": [weather_tool, search_tool]
    },
    {
        "query": "Search for Python tutorials",
        "expected_tool": "search_web",
        "expected_args": {"query": "Python tutorials"},
        "available_tools": [weather_tool, search_tool]
    }
]

results = test_tool_use_prompt(BASE_SYSTEM_PROMPT, test_cases)
print(f"Tool selection accuracy: {results['tool_selection_rate']:.1%}")
print(f"Argument accuracy: {results['argument_accuracy_rate']:.1%}")
\`\`\`

## Advanced: Chain-of-Thought for Tool Use

Combine CoT with tool use:

\`\`\`python
TOOL_COT_PROMPT = """When deciding whether and which tool to use, think step by step:

Step 1: Understand the query
- What is the user asking for?
- What type of information or action do they need?

Step 2: Check your knowledge
- Can I answer this from my training data?
- Is current/live data needed?
- Do I need to perform an action?

Step 3: Choose tool (if needed)
- Which tool best matches this need?
- Do I have all required parameters?
- Are there any constraints to consider?

Step 4: Execute
- Call the tool with correct parameters
- OR respond directly if no tool needed

Example:
Query: "What\'s the population of the capital of Japan?"

Step 1: User wants population of Japan's capital
Step 2: I know Tokyo is the capital, but population changes - need current data
Step 3: search_web would work, but query_database might have census data
Step 4: Try query_database first for authoritative data

Think through these steps before each response.
"""
\`\`\`

## Production Prompt Template

Complete production-ready template:

\`\`\`python
def create_tool_use_system_prompt(
    tools: List[Tool],
    user: User,
    constraints: Dict = None
) -> str:
    """
    Create a comprehensive system prompt for tool use.
    """
    
    # Base instructions
    prompt = """You are a helpful AI assistant with access to various tools.
Your goal is to help users by providing accurate, current information and performing actions on their behalf.
"""
    
    # List available tools
    prompt += "\\n\\nAvailable Tools:\\n"
    for tool in tools:
        prompt += f"- {tool.name}: {tool.description}\\n"
    
    # Usage guidelines
    prompt += "\\n\\nTool Usage Guidelines:\\n"
    prompt += "- Use tools when you need current information or to perform actions\\n"
    prompt += "- Answer directly from knowledge when appropriate\\n"
    prompt += "- Choose the most specific tool for the task\\n"
    prompt += "- Validate tool results before using them\\n"
    prompt += "- Handle errors gracefully and retry when appropriate\\n"
    
    # User-specific constraints
    if not user.is_admin:
        prompt += "\\n\\nNote: Some administrative tools are not available to you.\\n"
    
    # Custom constraints
    if constraints:
        prompt += "\\n\\nAdditional Constraints:\\n"
        for key, value in constraints.items():
            prompt += f"- {key}: {value}\\n"
    
    # Examples
    prompt += "\\n\\nExamples:\\n"
    prompt += "User: "What's the weather?" → Use get_weather\\n"
    prompt += "User: 'What is Python?' → Answer directly (general knowledge)\\n"
    prompt += "User: 'How many users signed up today?' → Use query_database\\n"
    
    return prompt

# Usage
system_prompt = create_tool_use_system_prompt(
    tools=registry.get_all(),
    user=current_user,
    constraints={"max_tool_calls": 5, "confirm_before_email": True}
)
\`\`\`

## Summary

Effective tool use prompting requires:

1. **Clear system prompts** explaining when/how to use tools
2. **Explicit guidelines** for tool selection
3. **Examples** of correct usage
4. **Parameter templates** for correct argument formatting
5. **Error handling** instructions
6. **Constraints** and boundaries
7. **Result processing** guidance
8. **Testing** to validate effectiveness

Remember: Your prompts are how the LLM learns to use tools. Invest time in crafting them carefully.

Next, we'll explore designing structured tool responses that LLMs can easily understand and act upon.
`,
};
