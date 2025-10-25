export const definingFunctionsToolsQuiz = [
  {
    id: 'q1',
    question:
      'You are designing a function schema for a complex API that has many optional parameters with interdependencies (e.g., if parameter A is provided, parameter B becomes required). How would you design the schema to handle this complexity while keeping it understandable for the LLM?',
    sampleAnswer: `Handling parameter interdependencies requires careful schema design and clear documentation:

**Approach 1: Flatten with Clear Documentation**
\`\`\`python
{
    "name": "create_event",
    "description": """Create a calendar event.
    
    Parameter dependencies:
    - If 'recurrence_pattern' is provided, 'recurrence_end_date' is REQUIRED
    - If 'attendees' is provided, 'send_invites' determines if invites are sent
    - If 'location_type' is 'virtual', 'meeting_link' is REQUIRED
    - If 'reminder' is true, 'reminder_minutes' must be provided
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "start_time": {"type": "string"},
            "recurrence_pattern": {
                "type": "string",
                "enum": ["daily", "weekly", "monthly",],
                "description": "If provided, recurrence_end_date is required"
            },
            "recurrence_end_date": {
                "type": "string",
                "description": "REQUIRED when recurrence_pattern is set"
            }
        },
        "required": ["title", "start_time",]
    }
}
\`\`\`

**Approach 2: Nested Objects for Related Parameters**
\`\`\`python
{
    "properties": {
        "title": {"type": "string"},
        "recurrence": {
            "type": "object",
            "description": "Optional recurrence settings. If provided, both fields required",
            "properties": {
                "pattern": {"type": "string", "enum": ["daily", "weekly",]},
                "end_date": {"type": "string"}
            },
            "required": ["pattern", "end_date",]
        },
        "location": {
            "type": "object",
            "description": "Location details",
            "properties": {
                "type": {"type": "string", "enum": ["physical", "virtual",]},
                "details": {"type": "string", "description": "Address or meeting link"}
            },
            "required": ["type", "details",]
        }
    }
}
\`\`\`

**Approach 3: Multiple Specialized Functions**
\`\`\`python
# Instead of one complex function, create variants
create_simple_event (title, start_time, location)
create_recurring_event (title, start_time, pattern, end_date)
create_virtual_event (title, start_time, meeting_link)
\`\`\`

**Best Practices:**
1. Document dependencies in description clearly
2. Use nested objects for tightly coupled parameters
3. Provide examples of valid combinations
4. Consider splitting into multiple simpler functions
5. Validate in function implementation, return helpful errors
6. Use Pydantic models for complex validation

**Validation Example:**
\`\`\`python
def create_event(**kwargs):
    # Validate dependencies
    if kwargs.get("recurrence_pattern") and not kwargs.get("recurrence_end_date"):
        return {
            "error": "recurrence_end_date is required when recurrence_pattern is set",
            "fix": "Please provide recurrence_end_date"
        }
    
    if kwargs.get("location_type") == "virtual" and not kwargs.get("meeting_link"):
        return {
            "error": "meeting_link is required for virtual events",
            "fix": "Please provide meeting_link"
        }
    
    # Proceed with event creation
\`\`\`

**Trade-offs:**
- **Flat schema**: Simpler for LLM, harder to enforce dependencies
- **Nested objects**: Enforces dependencies, but more complex
- **Multiple functions**: Clearest, but more functions to choose from
- **Description-based**: Relies on LLM understanding, needs good prompting

**Recommendation**: Use nested objects for strong dependencies, flat with clear descriptions for weak dependencies, and split into multiple functions when complexity is too high.`,
    keyPoints: [
      'Document parameter dependencies clearly in descriptions',
      'Use nested objects for tightly coupled parameters',
      'Consider splitting complex functions into simpler variants',
      'Implement validation with helpful error messages',
    ],
  },
  {
    id: 'q2',
    question:
      'Analyze the trade-offs between auto-generating function schemas from Python type hints versus hand-crafting schemas. In what scenarios would you choose each approach, and how would you handle edge cases like union types, optional parameters, and complex nested structures?',
    sampleAnswer: `Both approaches have distinct advantages and use cases:

**Auto-Generation Pros:**
1. **DRY Principle**: Single source of truth (Python function)
2. **Maintainability**: Schema updates automatically with code changes
3. **Speed**: Faster development for simple cases
4. **Consistency**: Reduces human error in schema writing
5. **Type Safety**: Leverages Python\'s type system

**Auto-Generation Cons:**
1. **Limited Control**: Hard to add LLM-specific guidance
2. **Description Quality**: Type hints don't capture semantic meaning
3. **Complex Types**: Union types, Literal, Optional need special handling
4. **Examples**: Can't easily add example values
5. **LLM Optimization**: Can't tailor for LLM understanding

**Hand-Crafting Pros:**
1. **Full Control**: Optimize descriptions for LLM comprehension
2. **Rich Documentation**: Add examples, constraints, use cases
3. **LLM-Friendly**: Structure for LLM understanding, not just type correctness
4. **Edge Cases**: Handle complex scenarios explicitly
5. **Evolution**: Can improve based on LLM usage patterns

**Hand-Crafting Cons:**
1. **Maintenance Burden**: Two sources of truth to keep in sync
2. **Verbose**: More code to write and maintain
3. **Error-Prone**: Manual schema can drift from implementation
4. **Slower**: Takes longer to create each function

**Hybrid Approach (Best of Both):**
\`\`\`python
from typing import Optional, Union, Literal
from pydantic import BaseModel, Field

class SearchParams(BaseModel):
    """Auto-generated base with manual enhancements."""
    query: str = Field(
        description="Search query. Examples: 'Python tutorials', 'machine learning basics'"
    )
    category: Optional[Literal["web", "images", "news",]] = Field(
        default="web",
        description="Type of search. Default: web"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results. Must be 1-100. Default: 10"
    )
    filters: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional filters. Example: {'language': 'en', 'date': 'recent'}"
    )

def generate_enhanced_schema (model: BaseModel) -> dict:
    """Generate schema with enhancements."""
    schema = model.schema()
    
    # Add LLM-friendly examples
    schema["examples",] = [
        {"query": "Python tutorials", "category": "web", "limit": 5},
        {"query": "cat videos", "category": "images"}
    ]
    
    # Add usage hints
    schema["usage_hints",] = "Use this for searching the web. Prefer specific queries."
    
    return schema
\`\`\`

**Handling Edge Cases:**

1. **Union Types:**
\`\`\`python
# Auto-generated:
location: Union[str, Tuple[float, float]]

# Hand-crafted description:
"location: City name ('Tokyo') OR coordinates as [lat, lon] (35.6762, 139.6503)"
\`\`\`

2. **Optional Parameters:**
\`\`\`python
def search (query: str, filters: Optional[Dict] = None):
    """
    query: REQUIRED - What to search for
    filters: OPTIONAL - Additional filters like {'date': 'recent'}
    """
\`\`\`

3. **Complex Nested:**
\`\`\`python
# Use Pydantic models for structure, manual descriptions for clarity
class Address(BaseModel):
    """Address information. Provide as much detail as available."""
    street: str = Field (description="Street address with number")
    city: str = Field (description="City name")
    country: str = Field (description="Country name or 2-letter code")
\`\`\`

**Decision Matrix:**

Use **Auto-Generation** when:
- Simple functions with straightforward parameters
- Internal tools (not user-facing)
- Rapid prototyping
- Strong type hints already exist
- Team has good Python typing discipline

Use **Hand-Crafting** when:
- Complex LLM interactions
- User-facing agents
- Need rich examples and edge case handling
- LLM struggle with auto-generated schemas
- Parameters have subtle requirements

Use **Hybrid** when:
- Production systems (most cases)
- Want speed of auto-generation + quality of manual
- Large codebase with many tools
- Need to iterate quickly while maintaining quality

**Best Practice**: Start with auto-generation, enhance manually where LLM struggles, and build a library of patterns for common cases.`,
    keyPoints: [
      'Auto-generation provides speed and consistency, hand-crafting offers control',
      'Use hybrid approach with Pydantic for best results',
      'Start with auto-generation, enhance manually where needed',
      'Document edge cases and complex types clearly',
    ],
  },
  {
    id: 'q3',
    question:
      'Create a comprehensive testing strategy for function schemas. How would you test that LLMs can correctly understand and use your schemas across different scenarios, edge cases, and with various prompting strategies?',
    sampleAnswer: `A comprehensive testing strategy for function schemas requires multiple layers of validation:

**1. Schema Validation Tests**
\`\`\`python
def test_schema_structure():
    """Test schema is valid JSON Schema."""
    schema = get_weather_schema()
    
    # Validate against JSON Schema spec
    validate_schema (schema)
    
    # Check required fields
    assert "name" in schema
    assert "description" in schema
    assert "parameters" in schema
    assert "properties" in schema["parameters",]

def test_schema_completeness():
    """Test schema has good descriptions."""
    schema = get_weather_schema()
    
    for param in schema["parameters",]["properties",]:
        assert len (param["description",]) > 20  # Non-trivial description
        assert "example" in param.lower() or has_enum (param)
\`\`\`

**2. LLM Understanding Tests**
\`\`\`python
def test_llm_can_call_function():
    """Test LLM generates correct function call."""
    test_cases = [
        {
            "prompt": "What\'s the weather in Tokyo?",
            "expected_function": "get_weather",
            "expected_args": {"location": "Tokyo"},
        },
        {
            "prompt": "Show me London weather in celsius",
            "expected_function": "get_weather",
            "expected_args": {"location": "London", "unit": "celsius"},
        }
    ]
    
    for case in test_cases:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": case["prompt",]}],
            functions=[weather_schema]
        )
        
        assert response.choices[0].message.function_call
        assert response.choices[0].message.function_call.name == case["expected_function",]
        
        args = json.loads (response.choices[0].message.function_call.arguments)
        for key, value in case["expected_args",].items():
            assert key in args
            assert args[key] == value

**3. Edge Case Tests**
\`\`\`python
def test_ambiguous_inputs():
    """Test LLM handles ambiguous inputs."""
    test_cases = [
        "What's the weather?"  # Missing location
        "Weather in SF"  # Ambiguous city
        "Show me weather tomorrow"  # Future date not supported
    ]
    
    for prompt in test_cases:
        response = call_llm (prompt, [weather_schema])
        
        # LLM should either ask for clarification or make reasonable assumption
        assert response is not None

def test_invalid_parameters():
    """Test schema prevents invalid parameters."""
    invalid_cases = [
        {"location": 123},  # Wrong type
        {"location": "Tokyo", "unit": "kelvin"},  # Invalid enum
        {},  # Missing required field
    ]
    
    for args in invalid_cases:
        with pytest.raises(ValidationError):
            validate_function_call("get_weather", args, weather_schema)

**4. Cross-Model Testing**
\`\`\`python
@pytest.mark.parametrize("model", ["gpt-4", "gpt-3.5-turbo", "claude-3-opus",])
def test_schema_works_across_models (model):
    """Test schema works with different LLMs."""
    response = call_llm_with_model(
        model=model,
        prompt="What's the weather in Paris?",
        functions=[weather_schema]
    )
    
    assert response.function_call
    assert response.function_call.name == "get_weather"
    assert "Paris" in response.function_call.arguments

**5. Prompt Variation Tests**
\`\`\`python
def test_various_phrasings():
    """Test LLM understands different ways of asking."""
    phrasings = [
        "What\'s the weather in London?",
        "Tell me London's weather",
        "How's the weather looking in London",
        "London weather please",
        "Is it raining in London?",
    ]
    
    for prompt in phrasings:
        response = call_llm (prompt, [weather_schema])
        args = json.loads (response.function_call.arguments)
        assert "London" in args.get("location", "")

**6. Schema Quality Metrics**
\`\`\`python
def test_schema_quality_score():
    """Score schema quality."""
    schema = get_weather_schema()
    
    score = 0
    
    # Description length and quality
    if len (schema["description",]) > 100:
        score += 20
    if "example" in schema["description",].lower():
        score += 10
    
    # Parameter descriptions
    for param in schema["parameters",]["properties",].values():
        if len (param.get("description", "")) > 30:
            score += 5
        if param.get("enum"):
            score += 5
    
    # Has examples
    if "examples" in schema:
        score += 15
    
    assert score >= 70  # Quality threshold

**7. Integration Tests**
\`\`\`python
def test_end_to_end_flow():
    """Test complete flow from prompt to execution."""
    user_message = "What\'s the weather in San Francisco and New York?"
    
    # 1. LLM generates function calls
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": user_message}],
        functions=[weather_schema]
    )
    
    # 2. Execute function
    func_name = response.choices[0].message.function_call.name
    func_args = json.loads (response.choices[0].message.function_call.arguments)
    
    result = execute_function (func_name, func_args)
    
    # 3. LLM synthesizes response
    final_response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": user_message},
            response.choices[0].message,
            {"role": "function", "name": func_name, "content": json.dumps (result)}
        ]
    )
    
    answer = final_response.choices[0].message.content
    assert "San Francisco" in answer or "New York" in answer

**8. Performance Tests**
\`\`\`python
def test_schema_performance():
    """Test schema doesn't slow down responses."""
    import time
    
    start = time.time()
    
    for _ in range(10):
        response = call_llm("Weather in London", [weather_schema])
    
    elapsed = time.time() - start
    avg_time = elapsed / 10
    
    assert avg_time < 2.0  # Average response under 2 seconds

**9. Monitoring in Production**
\`\`\`python
class SchemaMonitor:
    """Monitor schema usage in production."""
    
    def __init__(self):
        self.call_counts = defaultdict (int)
        self.errors = []
        self.success_rate = {}
    
    def record_call (self, function_name, success, error=None):
        self.call_counts[function_name] += 1
        if not success:
            self.errors.append({"function": function_name, "error": error})
    
    def get_report (self):
        return {
            "total_calls": sum (self.call_counts.values()),
            "error_rate": len (self.errors) / sum (self.call_counts.values()),
            "problematic_functions": [
                func for func, count in self.call_counts.items()
                if error_rate (func) > 0.1
            ]
        }
\`\`\`

**Continuous Improvement:**
1. Collect real user queries that fail
2. Add them to test suite
3. Improve schema based on failures
4. A/B test schema changes
5. Monitor success rates
6. Iterate on descriptions

**Best Practices:**
- Test with real user prompts, not just ideal cases
- Use multiple LLM models for validation
- Automate testing in CI/CD
- Track metrics over time
- Have human evaluation for subjective quality
- Test edge cases extensively
- Monitor production usage patterns`,
    keyPoints: [
      'Test schemas at multiple layers from structure to LLM understanding',
      'Use comprehensive test suites with edge cases and variations',
      'Monitor production usage and iterate based on failures',
      'Test across different models and prompting strategies',
    ],
  },
];
