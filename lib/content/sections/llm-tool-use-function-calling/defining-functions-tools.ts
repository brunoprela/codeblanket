export const definingFunctionsTools = {
  title: 'Defining Functions & Tools',
  id: 'defining-functions-tools',
  description:
    'Learn how to define functions and tools with proper schemas, type systems, and documentation for reliable LLM consumption.',
  content: `

# Defining Functions & Tools

## Introduction

The quality of your function definitions directly determines how reliably an LLM can use your tools. A well-defined function with a clear schema, detailed descriptions, and appropriate constraints will be called correctly almost every time. A poorly defined function leads to errors, hallucinations, and frustrated users.

In this section, we'll master the art of defining functions that LLMs can understand and use effectively. We'll explore JSON Schema, parameter types, descriptions, enums, and advanced patterns for complex tools.

## JSON Schema Fundamentals

Function definitions use JSON Schema, a standard for describing JSON data structures. Understanding JSON Schema is crucial for building reliable tools.

### Basic Schema Structure

\`\`\`python
function_schema = {
    "name": "function_name",           # Identifier for the function
    "description": "What it does",     # How the LLM decides when to use it
    "parameters": {                    # JSON Schema for parameters
        "type": "object",
        "properties": {
            # Parameter definitions go here
        },
        "required": ["param1", "param2"]  # Which parameters are required
    }
}
\`\`\`

### Parameter Types

JSON Schema supports these primitive types:

\`\`\`python
{
    "type": "object",
    "properties": {
        # String
        "name": {
            "type": "string",
            "description": "User's full name"
        },
        
        # Number (integer or float)
        "age": {
            "type": "integer",
            "description": "Age in years"
        },
        
        "price": {
            "type": "number",
            "description": "Price in dollars (can have decimals)"
        },
        
        # Boolean
        "is_active": {
            "type": "boolean",
            "description": "Whether the account is active"
        },
        
        # Array
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of tags"
        },
        
        # Object (nested)
        "address": {
            "type": "object",
            "properties": {
                "street": {"type": "string"},
                "city": {"type": "string"},
                "zip": {"type": "string"}
            },
            "description": "Mailing address"
        }
    }
}
\`\`\`

## Writing Excellent Descriptions

The description is **the most important part** of your schema. It's how the LLM decides whether and how to use your function.

### Function-Level Descriptions

\`\`\`python
# ❌ BAD: Too vague
{
    "name": "get_user",
    "description": "Gets a user"
}

# ❌ BAD: Missing critical details
{
    "name": "send_email",
    "description": "Sends an email"
}

# ✅ GOOD: Clear, detailed, specific
{
    "name": "get_user_profile",
    "description": """
    Retrieves detailed profile information for a specific user by their ID.
    Returns user's name, email, account type, registration date, and preferences.
    Use this when you need to look up information about a specific user.
    Do NOT use this for searching users - use search_users instead.
    """.strip()
}

# ✅ GOOD: Explains when to use and when NOT to use
{
    "name": "send_email",
    "description": """
    Sends an email to one or more recipients with optional attachments.
    Use this to actually send emails to users, not for drafting or previewing.
    Requires sender approval for production use.
    Maximum 100 recipients per call.
    """.strip()
}
\`\`\`

### Parameter Descriptions

Each parameter should explain:
1. What it is
2. What format it expects
3. Examples of valid values
4. Any constraints or special considerations

\`\`\`python
{
    "type": "object",
    "properties": {
        "user_id": {
            "type": "string",
            "description": "The unique identifier for the user. Format: 'usr_' followed by 24 alphanumeric characters (e.g., 'usr_abc123def456ghi789jkl')"
        },
        "date": {
            "type": "string",
            "description": "Date in ISO 8601 format (YYYY-MM-DD). Example: '2024-01-15'. Can also include time: '2024-01-15T14:30:00Z'"
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results to return. Must be between 1 and 100. Default: 20"
        },
        "sort_by": {
            "type": "string",
            "enum": ["date", "name", "relevance"],
            "description": "How to sort results. 'date' = newest first, 'name' = alphabetical, 'relevance' = most relevant first"
        }
    }
}
\`\`\`

## Using Enums for Constrained Values

Enums ensure the LLM can only choose from valid options:

\`\`\`python
{
    "name": "get_weather",
    "description": "Get current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name or ZIP code"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit", "kelvin"],
                "description": "Temperature unit. Defaults to celsius if not specified."
            },
            "include": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["temperature", "humidity", "wind", "pressure", "forecast"]
                },
                "description": "Additional data to include in the response"
            }
        },
        "required": ["location"]
    }
}
\`\`\`

Benefits:
- Prevents invalid values
- Makes options clear to the LLM
- Reduces errors
- Self-documenting

## Required vs Optional Parameters

Mark parameters as required when they're essential:

\`\`\`python
{
    "type": "object",
    "properties": {
        # Required: function cannot work without these
        "user_id": {
            "type": "string",
            "description": "User ID to retrieve"
        },
        
        # Optional: have sensible defaults
        "include_deleted": {
            "type": "boolean",
            "description": "Include deleted users in results. Default: false"
        },
        "limit": {
            "type": "integer",
            "description": "Max results. Default: 20"
        }
    },
    "required": ["user_id"]  # Only user_id is required
}
\`\`\`

Your Python function should handle optional parameters with defaults:

\`\`\`python
def get_user(user_id: str, include_deleted: bool = False, limit: int = 20):
    """Get user by ID."""
    # Implementation
    pass
\`\`\`

## Complex Parameter Types

### Arrays

\`\`\`python
{
    "recipients": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Email addresses of recipients",
        "minItems": 1,
        "maxItems": 100
    },
    "tags": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"}
            }
        },
        "description": "Key-value tag pairs"
    }
}
\`\`\`

### Nested Objects

\`\`\`python
{
    "filter": {
        "type": "object",
        "properties": {
            "date_range": {
                "type": "object",
                "properties": {
                    "start": {"type": "string", "description": "Start date (ISO 8601)"},
                    "end": {"type": "string", "description": "End date (ISO 8601)"}
                },
                "required": ["start", "end"]
            },
            "status": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["active", "pending", "completed", "cancelled"]
                }
            }
        },
        "description": "Filters to apply to the query"
    }
}
\`\`\`

### Union Types (anyOf)

When a parameter can be multiple types:

\`\`\`python
{
    "identifier": {
        "anyOf": [
            {"type": "string", "description": "User ID"},
            {"type": "integer", "description": "User number"}
        ],
        "description": "User identifier - can be either a string ID or integer number"
    }
}
\`\`\`

## Auto-Generating Schemas from Python

You can automatically generate schemas from Python functions using decorators:

\`\`\`python
from typing import Optional, List
from pydantic import BaseModel, Field
import inspect
import json

class FunctionSchema(BaseModel):
    """Pydantic model for function parameters."""
    pass

def generate_schema(func):
    """
    Generate JSON Schema from Python function signature and docstring.
    """
    sig = inspect.signature(func)
    doc = inspect.getdoc(func)
    
    # Parse docstring for description (simple example)
    description = doc.split('\\n')[0] if doc else ""
    
    # Build parameter schema
    properties = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        if param_name == 'self':
            continue
            
        param_type = param.annotation
        param_default = param.default
        
        # Map Python types to JSON Schema types
        if param_type == str:
            properties[param_name] = {"type": "string"}
        elif param_type == int:
            properties[param_name] = {"type": "integer"}
        elif param_type == float:
            properties[param_name] = {"type": "number"}
        elif param_type == bool:
            properties[param_name] = {"type": "boolean"}
        elif hasattr(param_type, '__origin__') and param_type.__origin__ == list:
            properties[param_name] = {
                "type": "array",
                "items": {"type": "string"}  # Simplified
            }
        else:
            properties[param_name] = {"type": "string"}  # Default
        
        # Check if required (no default value)
        if param_default == inspect.Parameter.empty:
            required.append(param_name)
    
    return {
        "name": func.__name__,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }

# Use it
def get_weather(location: str, unit: str = "celsius", include_forecast: bool = False):
    """Get current weather for a location."""
    pass

schema = generate_schema(get_weather)
print(json.dumps(schema, indent=2))
\`\`\`

Output:
\`\`\`json
{
  "name": "get_weather",
  "description": "Get current weather for a location.",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {"type": "string"},
      "unit": {"type": "string"},
      "include_forecast": {"type": "boolean"}
    },
    "required": ["location"]
  }
}
\`\`\`

## Using Pydantic for Schema Generation

Pydantic makes this much more robust:

\`\`\`python
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from enum import Enum

class TemperatureUnit(str, Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"
    KELVIN = "kelvin"

class GetWeatherParams(BaseModel):
    """Parameters for get_weather function."""
    location: str = Field(
        description="City name or ZIP code. Examples: 'San Francisco, CA' or '94102'"
    )
    unit: TemperatureUnit = Field(
        default=TemperatureUnit.CELSIUS,
        description="Temperature unit to use for the response"
    )
    include_forecast: bool = Field(
        default=False,
        description="Whether to include 7-day forecast in addition to current conditions"
    )

# Generate schema
schema = {
    "name": "get_weather",
    "description": "Get current weather and optional forecast for a location",
    "parameters": GetWeatherParams.schema()
}

print(json.dumps(schema, indent=2))
\`\`\`

Output:
\`\`\`json
{
  "name": "get_weather",
  "description": "Get current weather and optional forecast for a location",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City name or ZIP code. Examples: 'San Francisco, CA' or '94102'"
      },
      "unit": {
        "type": "string",
        "enum": ["celsius", "fahrenheit", "kelvin"],
        "default": "celsius",
        "description": "Temperature unit to use for the response"
      },
      "include_forecast": {
        "type": "boolean",
        "default": false,
        "description": "Whether to include 7-day forecast"
      }
    },
    "required": ["location"]
  }
}
\`\`\`

## Validation and Type Safety

Use Pydantic to validate LLM-generated arguments:

\`\`\`python
from pydantic import BaseModel, Field, validator
from typing import Optional
import json

class SendEmailParams(BaseModel):
    """Parameters for sending an email."""
    to: List[str] = Field(
        description="List of recipient email addresses",
        min_items=1,
        max_items=100
    )
    subject: str = Field(
        description="Email subject line",
        min_length=1,
        max_length=200
    )
    body: str = Field(
        description="Email body content (plain text or HTML)"
    )
    cc: Optional[List[str]] = Field(
        default=None,
        description="CC recipients"
    )
    priority: Literal["low", "normal", "high"] = Field(
        default="normal",
        description="Email priority level"
    )
    
    @validator('to', 'cc')
    def validate_emails(cls, v):
        """Validate email addresses."""
        if v is None:
            return v
        for email in v:
            if '@' not in email or '.' not in email:
                raise ValueError(f"Invalid email address: {email}")
        return v

def send_email(**kwargs):
    """Send an email with validation."""
    try:
        # Validate parameters
        params = SendEmailParams(**kwargs)
        
        # Send email using validated parameters
        print(f"Sending email to {params.to}")
        print(f"Subject: {params.subject}")
        print(f"Priority: {params.priority}")
        
        return {"success": True, "message_id": "msg_12345"}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

# Test with LLM-generated arguments
llm_args = {
    "to": ["user@example.com"],
    "subject": "Hello",
    "body": "This is a test email",
    "priority": "high"
}

result = send_email(**llm_args)
print(result)
\`\`\`

## Function Naming Conventions

Choose clear, descriptive function names:

### Good Practices

✅ **Verb-based names for actions:**
\`\`\`python
"send_email"
"create_user"
"delete_file"
"update_settings"
\`\`\`

✅ **Noun-based names for queries:**
\`\`\`python
"get_weather"
"search_products"
"list_users"
"fetch_report"
\`\`\`

✅ **Be specific:**
\`\`\`python
"get_current_weather"  # vs just "weather"
"send_email_with_attachments"  # vs just "email"
"search_products_by_category"  # vs just "search"
\`\`\`

❌ **Avoid ambiguous names:**
\`\`\`python
"do_thing"
"get_data"
"process"
"handle"
\`\`\`

## Documentation Patterns

### Inline Examples in Descriptions

\`\`\`python
{
    "date": {
        "type": "string",
        "description": """
        Date in ISO 8601 format.
        Examples: 
        - '2024-01-15' (date only)
        - '2024-01-15T14:30:00Z' (with time, UTC)
        - '2024-01-15T14:30:00-08:00' (with time and timezone)
        """
    }
}
\`\`\`

### Usage Guidelines

\`\`\`python
{
    "name": "search_products",
    "description": """
    Search the product catalog by keyword, category, or filters.
    
    Use this when:
    - User asks to find products
    - User wants to browse a category
    - User searches for specific features
    
    Do NOT use this when:
    - User has a specific product ID (use get_product instead)
    - User wants to see their order history (use get_orders)
    
    Returns up to 50 products ranked by relevance.
    """
}
\`\`\`

### Error Conditions

\`\`\`python
{
    "name": "get_user",
    "description": """
    Retrieve a user by their ID.
    
    Returns user object if found.
    Returns {"error": "User not found"} if user doesn't exist.
    Returns {"error": "Access denied"} if you don't have permission.
    """
}
\`\`\`

## Real-World Examples

### Weather API Tool

\`\`\`python
WEATHER_TOOL = {
    "name": "get_weather",
    "description": """
    Get current weather conditions and optional forecast for any location worldwide.
    
    Returns:
    - Current temperature and conditions
    - Humidity and wind speed
    - Optional 7-day forecast
    
    Use this when users ask about weather, temperature, or conditions.
    Works with city names, ZIP codes, or coordinates.
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "Location to get weather for. Can be: city name ('London'), city with country ('Paris, France'), ZIP code ('94102'), or coordinates ('37.7749,-122.4194')"
            },
            "units": {
                "type": "string",
                "enum": ["metric", "imperial"],
                "description": "Unit system. 'metric' for Celsius/km/h, 'imperial' for Fahrenheit/mph. Default: metric"
            },
            "include_forecast": {
                "type": "boolean",
                "description": "Include 7-day forecast. Default: false"
            },
            "language": {
                "type": "string",
                "description": "Language for weather descriptions (ISO 639-1 code, e.g., 'en', 'es', 'fr'). Default: en"
            }
        },
        "required": ["location"]
    }
}
\`\`\`

### Database Query Tool

\`\`\`python
DATABASE_QUERY_TOOL = {
    "name": "query_database",
    "description": """
    Execute a read-only SQL query against the production database.
    
    IMPORTANT: 
    - Only SELECT queries are allowed
    - Query timeout is 30 seconds
    - Maximum 1000 rows returned
    - Use for analytics and reporting only
    
    Available tables: users, orders, products, reviews
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "SQL SELECT query to execute. Must be read-only. Example: 'SELECT COUNT(*) FROM users WHERE created_at > '2024-01-01''"
            },
            "format": {
                "type": "string",
                "enum": ["json", "csv", "table"],
                "description": "Result format. Default: json"
            }
        },
        "required": ["query"]
    }
}
\`\`\`

### Email Tool

\`\`\`python
EMAIL_TOOL = {
    "name": "send_email",
    "description": """
    Send an email to one or more recipients.
    
    Features:
    - Multiple recipients (to, cc, bcc)
    - HTML or plain text
    - File attachments
    - Priority levels
    
    NOTE: All emails are logged and require sender approval in production.
    Maximum 100 recipients per email.
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "to": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Primary recipients (email addresses)",
                "minItems": 1,
                "maxItems": 100
            },
            "subject": {
                "type": "string",
                "description": "Email subject line (max 200 characters)",
                "maxLength": 200
            },
            "body": {
                "type": "string",
                "description": "Email body content"
            },
            "body_type": {
                "type": "string",
                "enum": ["text", "html"],
                "description": "Body content type. Default: text"
            },
            "cc": {
                "type": "array",
                "items": {"type": "string"},
                "description": "CC recipients (optional)"
            },
            "priority": {
                "type": "string",
                "enum": ["low", "normal", "high"],
                "description": "Email priority. Default: normal"
            },
            "attachments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "url": {"type": "string"}
                    }
                },
                "description": "File attachments (max 10 files, 25MB total)"
            }
        },
        "required": ["to", "subject", "body"]
    }
}
\`\`\`

## Testing Your Function Definitions

\`\`\`python
import openai
import json

def test_function_schema(function_schema, test_prompts):
    """
    Test if LLM can correctly use your function definition.
    """
    print(f"Testing function: {function_schema['name']}\\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"Test {i}: {prompt}")
        
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            functions=[function_schema],
            function_call={"name": function_schema["name"]}
        )
        
        if response.choices[0].message.function_call:
            args = json.loads(response.choices[0].message.function_call.arguments)
            print(f"✅ Generated args: {json.dumps(args, indent=2)}")
        else:
            print(f"❌ No function call generated")
        
        print()

# Test the weather tool
test_prompts = [
    "What's the weather in London?",
    "Show me the forecast for Tokyo in Celsius",
    "Weather in New York with 7-day forecast please"
]

test_function_schema(WEATHER_TOOL, test_prompts)
\`\`\`

## Common Pitfalls

### 1. Vague Descriptions
❌ \`"description": "Get data"\`
✅ \`"description": "Retrieve user profile data including name, email, and preferences"\`

### 2. Missing Examples
❌ \`"date": {"type": "string"}\`
✅ \`"date": {"type": "string", "description": "ISO 8601 date, e.g., '2024-01-15'"}\`

### 3. No Constraints
❌ \`"limit": {"type": "integer"}\`
✅ \`"limit": {"type": "integer", "minimum": 1, "maximum": 100}\`

### 4. Ambiguous Parameter Names
❌ \`"data", "info", "value"\`
✅ \`"user_email", "order_total", "product_sku"\`

### 5. Too Many Required Parameters
Make parameters optional when reasonable defaults exist.

## Summary

Great function definitions have:
1. **Clear, descriptive names** that indicate purpose
2. **Detailed descriptions** explaining when and how to use the function
3. **Well-documented parameters** with types, examples, and constraints
4. **Appropriate required/optional** parameter designation
5. **Enums** for constrained values
6. **Validation** using tools like Pydantic
7. **Examples** in descriptions
8. **Error condition documentation**

Remember: The LLM only knows about your function through its schema. Make it comprehensive!

Next, we'll explore different workflow patterns for orchestrating function calls.
`,
};
