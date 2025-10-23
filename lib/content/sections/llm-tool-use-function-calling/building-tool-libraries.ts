export const buildingToolLibraries = {
  title: 'Building Tool Libraries',
  id: 'building-tool-libraries',
  description:
    'Create reusable, well-organized libraries of tools that can be composed, discovered, and maintained effectively.',
  content: `

# Building Tool Libraries

## Introduction

As your agentic system grows, you'll accumulate dozens or even hundreds of tools. Without proper organization, this becomes unmanageable - tools are hard to find, duplicate functionality emerges, and maintenance becomes a nightmare. 

A well-designed tool library is organized, discoverable, composable, testable, and easy to extend. In this section, we'll learn how to build production-grade tool libraries that scale from a handful of tools to hundreds.

## Tool Library Architecture

A good tool library has several layers:

\`\`\`
Tool Library Architecture
├── Tools (Individual functions)
├── Tool Registry (Central catalog)
├── Tool Categories (Organization)
├── Tool Decorators (Metadata & validation)
├── Tool Loader (Dynamic loading)
└── Tool Documentation (Auto-generated)
\`\`\`

Let's build this bottom-up.

## Defining Individual Tools

Each tool should be self-contained and well-documented:

\`\`\`python
from typing import Any, Dict
from dataclasses import dataclass
from enum import Enum

class ToolCategory(Enum):
    """Categories for organizing tools."""
    WEB = "web"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    COMMUNICATION = "communication"
    COMPUTATION = "computation"
    API_INTEGRATION = "api_integration"

@dataclass
class Tool:
    """Represents a single tool."""
    name: str
    description: str
    function: callable
    schema: Dict[str, Any]
    category: ToolCategory
    requires_auth: bool = False
    requires_approval: bool = False
    estimated_cost: float = 0.0  # in dollars
    average_latency: float = 0.0  # in seconds
    
    def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments."""
        return self.function(**kwargs)
    
    def validate_args(self, **kwargs) -> bool:
        """Validate arguments against schema."""
        # Implementation
        pass

# Example tool definition
def get_weather_impl(location: str, unit: str = "celsius") -> Dict[str, Any]:
    """
    Implementation of weather fetching.
    """
    # Actual implementation here
    return {
        "location": location,
        "temperature": 72,
        "unit": unit,
        "conditions": "sunny"
    }

# Create tool with full metadata
weather_tool = Tool(
    name="get_weather",
    description="Get current weather conditions for any location worldwide",
    function=get_weather_impl,
    schema={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name or coordinates"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit"
            }
        },
        "required": ["location"]
    },
    category=ToolCategory.API_INTEGRATION,
    requires_auth=True,
    estimated_cost=0.001,
    average_latency=0.5
)
\`\`\`

## Tool Decorators

Use decorators to automatically generate tool metadata:

\`\`\`python
from functools import wraps
import inspect
from typing import get_type_hints

def tool(name: str = None, 
         description: str = None,
         category: ToolCategory = ToolCategory.COMPUTATION,
         requires_auth: bool = False,
         requires_approval: bool = False):
    """
    Decorator to mark a function as a tool and auto-generate schema.
    """
    def decorator(func):
        # Generate name from function if not provided
        tool_name = name or func.__name__
        
        # Get description from docstring if not provided
        tool_description = description or (inspect.getdoc(func) or "").split('\\n')[0]
        
        # Generate schema from type hints
        schema = generate_schema_from_function(func)
        
        # Store metadata on function
        func._tool_metadata = {
            "name": tool_name,
            "description": tool_description,
            "schema": schema,
            "category": category,
            "requires_auth": requires_auth,
            "requires_approval": requires_approval
        }
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

def generate_schema_from_function(func) -> Dict[str, Any]:
    """
    Generate JSON Schema from function signature and type hints.
    """
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)
    
    properties = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        if param_name in ['self', 'cls']:
            continue
        
        # Get type hint
        param_type = type_hints.get(param_name, str)
        
        # Map Python type to JSON Schema type
        json_type = python_type_to_json_type(param_type)
        
        properties[param_name] = {
            "type": json_type,
            "description": f"Parameter {param_name}"
        }
        
        # Check if required (no default value)
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
    
    return {
        "type": "object",
        "properties": properties,
        "required": required
    }

def python_type_to_json_type(python_type) -> str:
    """Map Python types to JSON Schema types."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object"
    }
    return type_map.get(python_type, "string")

# Usage
@tool(
    description="Get current weather for a location",
    category=ToolCategory.API_INTEGRATION,
    requires_auth=True
)
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Get weather information for a specific location."""
    # Implementation
    return {"temperature": 72, "conditions": "sunny"}

# Access metadata
print(get_weather._tool_metadata)
\`\`\`

## Tool Registry

Centralized registry for all tools:

\`\`\`python
from typing import List, Optional, Dict, Any
import logging

class ToolRegistry:
    """
    Central registry for all available tools.
    """
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[ToolCategory, List[str]] = {
            category: [] for category in ToolCategory
        }
        self.logger = logging.getLogger(__name__)
    
    def register(self, tool: Tool):
        """Register a new tool."""
        if tool.name in self._tools:
            self.logger.warning(f"Tool {tool.name} already registered, overwriting")
        
        self._tools[tool.name] = tool
        self._categories[tool.category].append(tool.name)
        
        self.logger.info(f"Registered tool: {tool.name} ({tool.category.value})")
    
    def unregister(self, tool_name: str):
        """Unregister a tool."""
        if tool_name in self._tools:
            tool = self._tools[tool_name]
            self._categories[tool.category].remove(tool_name)
            del self._tools[tool_name]
            self.logger.info(f"Unregistered tool: {tool_name}")
    
    def get(self, tool_name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(tool_name)
    
    def get_all(self) -> List[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    def get_by_category(self, category: ToolCategory) -> List[Tool]:
        """Get all tools in a category."""
        tool_names = self._categories[category]
        return [self._tools[name] for name in tool_names]
    
    def search(self, query: str) -> List[Tool]:
        """Search tools by name or description."""
        query_lower = query.lower()
        results = []
        
        for tool in self._tools.values():
            if (query_lower in tool.name.lower() or 
                query_lower in tool.description.lower()):
                results.append(tool)
        
        return results
    
    def get_schemas(self, tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get OpenAI-compatible function schemas for specified tools.
        If tool_names is None, return all tool schemas.
        """
        if tool_names is None:
            tools = self.get_all()
        else:
            tools = [self.get(name) for name in tool_names if self.get(name)]
        
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.schema
            }
            for tool in tools
        ]
    
    def execute(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool by name."""
        tool = self.get(tool_name)
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")
        
        self.logger.info(f"Executing tool: {tool_name}")
        return tool.execute(**kwargs)

# Create global registry
registry = ToolRegistry()

# Register tools
registry.register(weather_tool)
registry.register(search_tool)
registry.register(email_tool)

# Use registry
all_tools = registry.get_all()
web_tools = registry.get_by_category(ToolCategory.WEB)
weather_results = registry.search("weather")
\`\`\`

## Auto-Registration from Modules

Automatically discover and register tools:

\`\`\`python
import importlib
import os
from pathlib import Path

def auto_register_tools(registry: ToolRegistry, package_path: str):
    """
    Automatically discover and register all tools in a package.
    
    Looks for functions decorated with @tool in all Python files.
    """
    package_dir = Path(package_path)
    
    for py_file in package_dir.rglob("*.py"):
        if py_file.name.startswith("_"):
            continue
        
        # Convert file path to module name
        relative_path = py_file.relative_to(package_dir.parent)
        module_name = str(relative_path.with_suffix("")).replace(os.sep, ".")
        
        try:
            # Import module
            module = importlib.import_module(module_name)
            
            # Find all functions with tool metadata
            for name in dir(module):
                obj = getattr(module, name)
                if callable(obj) and hasattr(obj, "_tool_metadata"):
                    # Create Tool instance
                    metadata = obj._tool_metadata
                    tool = Tool(
                        name=metadata["name"],
                        description=metadata["description"],
                        function=obj,
                        schema=metadata["schema"],
                        category=metadata.get("category", ToolCategory.COMPUTATION),
                        requires_auth=metadata.get("requires_auth", False),
                        requires_approval=metadata.get("requires_approval", False)
                    )
                    registry.register(tool)
        
        except Exception as e:
            logging.error(f"Failed to load tools from {py_file}: {e}")

# Usage
auto_register_tools(registry, "tools/")
\`\`\`

## Tool Categories and Organization

Organize tools into logical categories:

\`\`\`
tools/
├── __init__.py
├── web/
│   ├── __init__.py
│   ├── search.py
│   ├── scrape.py
│   └── browser.py
├── database/
│   ├── __init__.py
│   ├── query.py
│   └── update.py
├── communication/
│   ├── __init__.py
│   ├── email.py
│   ├── slack.py
│   └── sms.py
└── computation/
    ├── __init__.py
    ├── math.py
    └── data_analysis.py
\`\`\`

Each module contains related tools:

\`\`\`python
# tools/web/search.py
from tools.decorators import tool
from tools.categories import ToolCategory

@tool(
    description="Search the web using Google",
    category=ToolCategory.WEB
)
def google_search(query: str, num_results: int = 10) -> dict:
    """Search Google and return top results."""
    # Implementation
    pass

@tool(
    description="Search Wikipedia for articles",
    category=ToolCategory.WEB
)
def wikipedia_search(query: str) -> dict:
    """Search Wikipedia."""
    # Implementation
    pass
\`\`\`

## Tool Composition

Create higher-level tools by composing simpler ones:

\`\`\`python
@tool(
    description="Research a topic by searching multiple sources",
    category=ToolCategory.WEB
)
def research_topic(topic: str) -> dict:
    """
    Research a topic comprehensively.
    
    Searches Google, Wikipedia, and academic sources,
    then synthesizes the results.
    """
    # Use other tools
    google_results = google_search(query=topic, num_results=5)
    wiki_results = wikipedia_search(query=topic)
    scholar_results = google_scholar_search(query=topic)
    
    return {
        "topic": topic,
        "google": google_results,
        "wikipedia": wiki_results,
        "scholar": scholar_results,
        "summary": synthesize_research([
            google_results,
            wiki_results,
            scholar_results
        ])
    }

def synthesize_research(results: list) -> str:
    """Synthesize research results using LLM."""
    # Use LLM to create summary
    pass
\`\`\`

## Tool Versioning

Support multiple versions of tools:

\`\`\`python
@tool(
    name="send_email_v1",
    description="Send email (v1 - basic)",
    category=ToolCategory.COMMUNICATION
)
def send_email_v1(to: str, subject: str, body: str) -> dict:
    """Original email function."""
    pass

@tool(
    name="send_email_v2",
    description="Send email (v2 - with attachments)",
    category=ToolCategory.COMMUNICATION
)
def send_email_v2(to: str, subject: str, body: str, 
                  attachments: list = None) -> dict:
    """Enhanced email function with attachments."""
    pass

# Registry supports aliasing
registry.register(send_email_v2)
registry.alias("send_email", "send_email_v2")  # Default to v2
\`\`\`

## Tool Documentation Generation

Auto-generate documentation from tool registry:

\`\`\`python
def generate_tool_documentation(registry: ToolRegistry, output_file: str):
    """
    Generate markdown documentation for all tools.
    """
    docs = ["# Tool Library Documentation\\n\\n"]
    
    # Group by category
    for category in ToolCategory:
        tools = registry.get_by_category(category)
        if not tools:
            continue
        
        docs.append(f"## {category.value.replace('_', ' ').title()}\\n\\n")
        
        for tool in sorted(tools, key=lambda t: t.name):
            docs.append(f"### \`{tool.name}\`\\n\\n")
            docs.append(f"{tool.description}\\n\\n")
            
            # Parameters
            docs.append("**Parameters:**\\n\\n")
            for param_name, param_info in tool.schema["properties"].items():
                required = " (required)" if param_name in tool.schema.get("required", []) else ""
                docs.append(f"- \`{param_name}\` ({param_info['type']}){required}: {param_info.get('description', '')}\\n")
            
            docs.append("\\n")
            
            # Metadata
            if tool.requires_auth:
                docs.append("⚠️ *Requires authentication*\\n\\n")
            if tool.requires_approval:
                docs.append("⚠️ *Requires user approval*\\n\\n")
            
            docs.append(f"**Estimated cost:** \${tool.estimated_cost:.4f}\\n\\n")
            docs.append(f"**Average latency:** {tool.average_latency:.2f}s\\n\\n")
docs.append("---\\n\\n")
    
    # Write to file
with open(output_file, 'w') as f:
f.write(''.join(docs))

# Generate documentation
generate_tool_documentation(registry, "docs/tools.md")
\`\`\`

## Tool Testing

Comprehensive testing for tools:

\`\`\`python
import pytest
from unittest.mock import Mock, patch

class ToolTestCase:
    """Base class for tool tests."""
    
    def test_tool_registered(self, registry):
        """Test that tool is registered."""
        tool = registry.get(self.tool_name)
        assert tool is not None
        assert tool.name == self.tool_name
    
    def test_tool_schema(self, registry):
        """Test that schema is valid."""
        tool = registry.get(self.tool_name)
        schema = tool.schema
        
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
    
    def test_tool_execution(self, registry):
        """Test that tool executes successfully."""
        tool = registry.get(self.tool_name)
        result = tool.execute(**self.test_params)
        assert result is not None

class TestWeatherTool(ToolTestCase):
    tool_name = "get_weather"
    test_params = {"location": "San Francisco", "unit": "celsius"}
    
    def test_weather_returns_temperature(self, registry):
        """Test that weather tool returns temperature."""
        tool = registry.get(self.tool_name)
        result = tool.execute(**self.test_params)
        
        assert "temperature" in result
        assert isinstance(result["temperature"], (int, float))
    
    @patch('tools.web.weather.requests.get')
    def test_weather_api_error(self, mock_get, registry):
        """Test error handling when API fails."""
        mock_get.side_effect = Exception("API Error")
        
        tool = registry.get(self.tool_name)
        result = tool.execute(**self.test_params)
        
        assert "error" in result

# Run tests
pytest tools/tests/
\`\`\`

## Tool Permissions and Security

Control access to sensitive tools:

\`\`\`python
from enum import Enum

class Permission(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"

class User:
    def __init__(self, user_id: str, permissions: List[Permission]):
        self.user_id = user_id
        self.permissions = permissions

class SecureToolRegistry(ToolRegistry):
    """Tool registry with permission checking."""
    
    def __init__(self):
        super().__init__()
        self._tool_permissions: Dict[str, List[Permission]] = {}
    
    def register(self, tool: Tool, required_permissions: List[Permission] = None):
        """Register tool with required permissions."""
        super().register(tool)
        if required_permissions:
            self._tool_permissions[tool.name] = required_permissions
    
    def can_execute(self, tool_name: str, user: User) -> bool:
        """Check if user has permission to execute tool."""
        required = self._tool_permissions.get(tool_name, [])
        return all(perm in user.permissions for perm in required)
    
    def execute(self, tool_name: str, user: User, **kwargs) -> Any:
        """Execute tool with permission check."""
        if not self.can_execute(tool_name, user):
            raise PermissionError(
                f"User {user.user_id} does not have permission to execute {tool_name}"
            )
        
        return super().execute(tool_name, **kwargs)

# Usage
secure_registry = SecureToolRegistry()

# Register tool with permissions
secure_registry.register(
    delete_file_tool,
    required_permissions=[Permission.WRITE, Permission.DELETE]
)

# Execute with user
user = User("user_123", permissions=[Permission.READ])
try:
    secure_registry.execute("delete_file", user, path="/important.txt")
except PermissionError as e:
    print(f"Access denied: {e}")
\`\`\`

## Tool Metrics and Monitoring

Track tool usage and performance:

\`\`\`python
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class ToolMetrics:
    """Metrics for a single tool."""
    tool_name: str
    execution_count: int = 0
    total_latency: float = 0.0
    error_count: int = 0
    last_execution: datetime = None
    
    @property
    def average_latency(self) -> float:
        if self.execution_count == 0:
            return 0.0
        return self.total_latency / self.execution_count
    
    @property
    def error_rate(self) -> float:
        if self.execution_count == 0:
            return 0.0
        return self.error_count / self.execution_count

class MonitoredToolRegistry(ToolRegistry):
    """Tool registry with metrics tracking."""
    
    def __init__(self):
        super().__init__()
        self._metrics: Dict[str, ToolMetrics] = defaultdict(
            lambda: ToolMetrics(tool_name="")
        )
    
    def execute(self, tool_name: str, **kwargs) -> Any:
        """Execute tool and track metrics."""
        start_time = time.time()
        metrics = self._metrics[tool_name]
        metrics.tool_name = tool_name
        metrics.execution_count += 1
        metrics.last_execution = datetime.now()
        
        try:
            result = super().execute(tool_name, **kwargs)
            return result
        except Exception as e:
            metrics.error_count += 1
            raise
        finally:
            elapsed = time.time() - start_time
            metrics.total_latency += elapsed
    
    def get_metrics(self, tool_name: str) -> ToolMetrics:
        """Get metrics for a specific tool."""
        return self._metrics[tool_name]
    
    def get_all_metrics(self) -> List[ToolMetrics]:
        """Get metrics for all tools."""
        return list(self._metrics.values())
    
    def print_metrics_report(self):
        """Print a metrics report."""
        print("Tool Metrics Report")
        print("=" * 80)
        print(f"{'Tool':<30} {'Executions':<12} {'Avg Latency':<15} {'Error Rate':<12}")
        print("-" * 80)
        
        for metrics in sorted(self._metrics.values(), 
                            key=lambda m: m.execution_count, 
                            reverse=True):
            print(f"{metrics.tool_name:<30} "
                  f"{metrics.execution_count:<12} "
                  f"{metrics.average_latency:<15.3f} "
                  f"{metrics.error_rate:<12.2%}")

# Usage
monitored_registry = MonitoredToolRegistry()
monitored_registry.register(weather_tool)

# Use tools
monitored_registry.execute("get_weather", location="SF")
monitored_registry.execute("get_weather", location="NYC")

# View metrics
monitored_registry.print_metrics_report()
\`\`\`

## Real-World Example: Complete Tool Library

\`\`\`python
# tools/__init__.py
from .registry import registry
from .decorators import tool
from .categories import ToolCategory

# Auto-register all tools
from .web import *
from .database import *
from .communication import *
from .computation import *

__all__ = ['registry', 'tool', 'ToolCategory']

# tools/web/search.py
from tools import tool, ToolCategory

@tool(category=ToolCategory.WEB)
def google_search(query: str, num_results: int = 10) -> dict:
    """Search Google and return top results."""
    # Implementation
    pass

@tool(category=ToolCategory.WEB)
def wikipedia_search(query: str) -> dict:
    """Search Wikipedia articles."""
    # Implementation
    pass

# main.py - Using the library
from tools import registry
import openai

# Get all tool schemas
schemas = registry.get_schemas()

# Use in LLM conversation
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Search for information about Python"}],
    functions=schemas
)

# Execute tool if called
if response.choices[0].message.function_call:
    func_name = response.choices[0].message.function_call.name
    func_args = json.loads(response.choices[0].message.function_call.arguments)
    
    result = registry.execute(func_name, **func_args)
\`\`\`

## Summary

A production-grade tool library needs:

1. **Structured organization** - Categories, modules, clear hierarchy
2. **Central registry** - Single source of truth for all tools
3. **Auto-registration** - Discover tools automatically
4. **Rich metadata** - Descriptions, categories, requirements
5. **Schema generation** - From type hints and decorators
6. **Versioning** - Support multiple versions of tools
7. **Testing** - Comprehensive test coverage
8. **Documentation** - Auto-generated and up-to-date
9. **Security** - Permission checking and access control
10. **Monitoring** - Track usage, performance, and errors

Next, we'll explore specific patterns for building common types of tools like API integrations.
`,
};
