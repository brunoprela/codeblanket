export const buildingToolLibrariesQuiz = [
  {
    id: 'q1',
    question:
      'Design a scalable tool registry system that can handle hundreds of tools across multiple categories. How would you implement tool discovery, version management, and dynamic loading while maintaining performance?',
    sampleAnswer: `A production tool registry needs efficient indexing, lazy loading, and versioning:

**Core Architecture:**
\`\`\`python
class ScalableToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._index = ToolIndex()  # For fast lookups
        self._cache = LRUCache(maxsize=100)
        self._loader = DynamicToolLoader()
    
    def register(self, tool: Tool):
        # Store tool
        self._tools[tool.name] = tool
        
        # Update indexes
        self._index.add(tool)
        
        # Clear relevant caches
        self._cache.invalidate(f"category:{tool.category}")
\`\`\`

**Fast Search with Indexing:**
\`\`\`python
class ToolIndex:
    def __init__(self):
        self.by_category = defaultdict(set)
        self.by_tag = defaultdict(set)
        self.search_index = {}  # For full-text search
    
    def search(self, query: str) -> List[str]:
        # Combine multiple index lookups
        results = set()
        
        # Category match
        if query in self.by_category:
            results.update(self.by_category[query])
        
        # Full-text search on descriptions
        words = query.lower().split()
        for word in words:
            if word in self.search_index:
                results.update(self.search_index[word])
        
        return list(results)
\`\`\`

**Version Management:**
\`\`\`python
class VersionedTool:
    def __init__(self, name: str, version: str, tool: Tool):
        self.name = name
        self.version = version
        self.tool = tool

class VersionedRegistry:
    def __init__(self):
        self.versions: Dict[str, Dict[str, Tool]] = {}
        self.aliases: Dict[str, str] = {}  # name -> default version
    
    def register(self, tool: Tool, version: str):
        if tool.name not in self.versions:
            self.versions[tool.name] = {}
        
        self.versions[tool.name][version] = tool
        
        # Set as default if first version or marked as default
        if version.endswith("@default"):
            self.aliases[tool.name] = version
    
    def get(self, name: str, version: str = None) -> Tool:
        if version is None:
            version = self.aliases.get(name, "latest")
        
        return self.versions.get(name, {}).get(version)
\`\`\`

**Dynamic Loading:**
\`\`\`python
class DynamicToolLoader:
    def __init__(self):
        self.module_cache = {}
    
    def load_tool(self, module_path: str) -> Tool:
        if module_path in self.module_cache:
            return self.module_cache[module_path]
        
        # Import module dynamically
        module = importlib.import_module(module_path)
        
        # Find tool in module
        for name, obj in inspect.getmembers(module):
            if hasattr(obj, '_tool_metadata'):
                tool = self._create_tool_from_metadata(obj)
                self.module_cache[module_path] = tool
                return tool
\`\`\`

**Performance Optimizations:**
1. Lazy loading - only load tools when needed
2. Caching - LRU cache for frequently used tools
3. Indexing - fast lookups by category, tags, keywords
4. Batch operations - register multiple tools efficiently
5. Parallel loading - load tools concurrently

**Best Practices:**
- Use semantic versioning (semver)
- Deprecation warnings for old versions
- Migration guides between versions
- Performance benchmarks per tool
- Auto-documentation generation`,
    keyPoints: [
      'Key concept from answer',
      'Key concept from answer',
      'Key concept from answer',
    ],
  },
  {
    id: 'q2',
    question:
      'How would you implement tool composition and higher-order tools that create or modify other tools? Provide examples of when this is useful and discuss the security implications.',
    sampleAnswer: `Tool composition and meta-tools enable powerful abstractions but require careful security design:

**Basic Composition:**
\`\`\`python
class CompositeTool(Tool):
    def __init__(self, name: str, tools: List[Tool]):
        self.tools = tools
        super().__init__(name, self._compose_description())
    
    def _compose_description(self):
        return f"Composite of: {', '.join(t.name for t in self.tools)}"
    
    def execute(self, **kwargs):
        results = []
        for tool in self.tools:
            result = tool.execute(**kwargs)
            results.append(result)
        return {"combined": results}

# Usage
research_tool = CompositeTool(
    "comprehensive_research",
    [google_search_tool, wikipedia_tool, scholar_tool]
)
\`\`\`

**Tool Factories:**
\`\`\`python
def create_http_tools_from_openapi(spec: Dict) -> List[Tool]:
    """Generate tools from OpenAPI specification."""
    tools = []
    
    for path, methods in spec["paths",].items():
        for method, details in methods.items():
            tool = Tool(
                name=f"{method}_{path.replace('/', '_')}",
                description=details.get("summary"),
                function=create_http_function(method, path),
                schema=extract_schema(details["parameters",])
            )
            tools.append(tool)
    
    return tools
\`\`\`

**Meta-Tools:**
\`\`\`python
@tool(description="Create a new tool from description")
def create_tool_dynamically(name: str, description: str, 
                           code: str) -> dict:
    # DANGER: Security implications!
    
    # 1. Validate code (syntax check, no dangerous imports)
    if not validate_code_safety(code):
        return {"error": "Unsafe code detected"}
    
    # 2. Execute in restricted environment
    namespace = create_restricted_namespace()
    exec(code, namespace)
    
    # 3. Extract function
    func = namespace.get(name)
    if not callable(func):
        return {"error": "No function found"}
    
    # 4. Create and register tool
    new_tool = Tool(
        name=name,
        description=description,
        function=func,
        schema=generate_schema_from_function(func)
    )
    
    registry.register(new_tool)
    
    return {"success": True, "tool": name}
\`\`\`

**Security Considerations:**

1. **Code Validation:**
\`\`\`python
def validate_code_safety(code: str) -> bool:
    # Parse code
    tree = ast.parse(code)
    
    # Check for dangerous patterns
    dangerous = [
        ast.Import,  # Block imports
        ast.Exec,  # Block exec
        ast.Eval,  # Block eval
    ]
    
    for node in ast.walk(tree):
        if isinstance(node, tuple(dangerous)):
            return False
    
    return True
\`\`\`

2. **Sandboxing:**
\`\`\`python
def create_restricted_namespace():
    # Only allow safe builtins
    safe_builtins = {
        'print': print,
        'len': len,
        'range': range,
        # ... safe functions only
    }
    
    return {'__builtins__': safe_builtins}
\`\`\`

3. **Permission System:**
\`\`\`python
class ToolPermissions:
    def can_create_tools(self, user: User) -> bool:
        return user.has_permission("tool.create")
    
    def can_modify_tools(self, user: User) -> bool:
        return user.has_permission("tool.modify")
\`\`\`

**Use Cases:**
1. **API Integration:** Auto-generate tools from API specs
2. **Custom Workflows:** Users create domain-specific tools
3. **Testing:** Generate mock tools for testing
4. **Optimization:** Create specialized tools from general ones
5. **Migration:** Transform tools between versions

**Best Practices:**
- Never execute untrusted code directly
- Use sandboxing (Docker, RestrictedPython)
- Audit all dynamically created tools
- Rate limit tool creation
- Require admin approval for new tools
- Log all tool modifications
- Have rollback capability`,
    keyPoints: [
      'Key concept from answer',
      'Key concept from answer',
      'Key concept from answer',
    ],
  },
  {
    id: 'q3',
    question:
      'Design a tool testing and validation framework that ensures tools work correctly, have good documentation, and perform well. Include unit tests, integration tests, and performance benchmarks.',
    sampleAnswer: `Comprehensive tool testing requires multiple layers:

**Test Pyramid:**
1. Unit Tests (fast, many)
2. Integration Tests (medium speed, moderate)
3. End-to-End Tests (slow, few)
4. Performance Tests (periodic)

**Unit Testing Framework:**
\`\`\`python
class ToolTestSuite:
    def __init__(self, tool: Tool):
        self.tool = tool
    
    def test_schema_valid(self):
        """Schema is valid JSON Schema."""
        validate_json_schema(self.tool.schema)
    
    def test_execute_basic(self):
        """Tool executes with valid input."""
        result = self.tool.execute(**self.get_valid_input())
        assert result is not None
    
    def test_error_handling(self):
        """Tool handles errors gracefully."""
        result = self.tool.execute(**self.get_invalid_input())
        assert "error" in result
    
    def test_documentation(self):
        """Tool has good documentation."""
        assert len(self.tool.description) > 50
        for param in self.tool.schema["properties",]:
            assert "description" in param
\`\`\`

**Integration Testing:**
\`\`\`python
class ToolIntegrationTests:
    def test_with_llm(self):
        """LLM can understand and use tool."""
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Use get_weather for Tokyo"}],
            functions=[self.tool.to_schema()]
        )
        
        assert response.choices[0].message.function_call
        assert response.choices[0].message.function_call.name == self.tool.name
    
    def test_with_real_data(self):
        """Tool works with production-like data."""
        result = self.tool.execute(**self.get_real_world_input())
        self.validate_output(result)
\`\`\`

**Performance Benchmarks:**
\`\`\`python
class ToolPerformanceBenchmark:
    def benchmark_latency(self, iterations=100):
        """Measure average latency."""
        times = []
        for _ in range(iterations):
            start = time.time()
            self.tool.execute(**self.get_input())
            times.append(time.time() - start)
        
        return {
            "avg_ms": statistics.mean(times) * 1000,
            "p95_ms": statistics.quantiles(times, n=20)[18] * 1000,
            "p99_ms": statistics.quantiles(times, n=100)[98] * 1000
        }
    
    def benchmark_throughput(self, duration=10):
        """Measure requests per second."""
        count = 0
        start = time.time()
        
        while time.time() - start < duration:
            self.tool.execute(**self.get_input())
            count += 1
        
        return count / duration
\`\`\`

**Automated Quality Checks:**
\`\`\`python
class ToolQualityValidator:
    def validate_all(self, tool: Tool) -> List[str]:
        issues = []
        
        # Documentation
        if len(tool.description) < 50:
            issues.append("Description too short")
        
        # Schema completeness
        for param in tool.schema["properties",].values():
            if "description" not in param:
                issues.append("Parameter missing description")
        
        # Performance
        latency = self.measure_latency(tool)
        if latency > 5000:  # 5 seconds
            issues.append(f"High latency: {latency}ms")
        
        # Error rate
        error_rate = self.measure_error_rate(tool)
        if error_rate > 0.05:  # 5%
            issues.append(f"High error rate: {error_rate:.1%}")
        
        return issues
\`\`\`

**CI/CD Integration:**
\`\`\`python
# pytest configuration
@pytest.fixture(scope="module")
def all_tools():
    return registry.get_all()

@pytest.mark.parametrize("tool", all_tools())
def test_tool_quality(tool):
    validator = ToolQualityValidator()
    issues = validator.validate_all(tool)
    assert len(issues) == 0, f"Tool {tool.name} has issues: {issues}"
\`\`\`

**Best Practices:**
- Test every tool automatically
- Run tests in CI/CD
- Performance regression testing
- Monitor tools in production
- A/B test tool changes
- Document test coverage
- Regular benchmarking`,
    keyPoints: [
      'Key concept from answer',
      'Key concept from answer',
      'Key concept from answer',
    ],
  },
];
