export const functionCallingFundamentalsQuiz = [
  {
    id: 'q1',
    question:
      'Explain the fundamental difference between function calling and traditional prompt-based information extraction. Why is function calling more reliable, and in what scenarios might traditional prompting still be preferable?',
    sampleAnswer: `Function calling provides a structured, schema-enforced way for LLMs to generate outputs, making it fundamentally more reliable than parsing text responses. Here\'s why:

**Key Differences:**1. **Structure vs. Text**: Function calling uses JSON schemas with defined types, while text parsing relies on pattern matching and hope
2. **Validation**: Function arguments are validated against schemas; text requires custom parsing logic
3. **Type Safety**: Function calling preserves types (strings, numbers, booleans); text is always strings
4. **Consistency**: Function calling guarantees format; text parsing deals with variations like "The city is SF" vs "SF" vs "San Francisco, CA"

**Why Function Calling is More Reliable:**
- LLM knows exactly what format to produce
- No ambiguity in parameter names or types
- Built-in validation catches errors before execution
- Easier error handling with predictable structure
- Reduces hallucination by constraining output format

**When Traditional Prompting is Better:**
- Free-form creative writing or explanations
- When you want the LLM's natural language synthesis
- Simple queries that don't need structured output
- When function schemas would be more complex than parsing
- For one-off tasks where setup time matters

**Real-World Example:**
Extracting a city name from "Show me weather for SF" - with function calling, you get {"location": "San Francisco"} reliably. With text parsing, you might get "SF", "San Francisco", "The location is SF", or variations requiring complex regex.

**Best Practice**: Use function calling when you need structured data or actions; use traditional prompting when you need natural language responses or explanations.`,
    keyPoints: [
      'Key concept from discussion question',
      'Key concept from discussion question',
      'Key concept from discussion question',
      'Key concept from discussion question',
    ],
  },
  {
    id: 'q2',
    question:
      'Design a comprehensive error handling strategy for a production function calling system. Consider network failures, rate limiting, invalid function calls, and timeout scenarios. How would you implement retries, fallbacks, and user feedback?',
    sampleAnswer: `A production function calling system needs multi-layered error handling to ensure reliability and good user experience:

**Error Categories and Strategies:**1. **Network/API Errors (Transient)**
   - Implement exponential backoff with jitter
   - Retry 3-5 times with delays: 1s, 2s, 4s, 8s
   - Add jitter (±50%) to prevent thundering herd
   - Circuit breaker after repeated failures
   - Fallback to cached responses if available

2. **Rate Limiting (429 Errors)**
   - Respect Retry-After headers
   - Implement token bucket rate limiting
   - Queue requests when limit approached
   - Provide user feedback: "API is busy, retrying..."
   - Consider backup providers (OpenAI → Anthropic)

3. **Invalid Function Calls**
   - Validate arguments before execution
   - Return clear error messages to LLM
   - Allow LLM to retry with corrected arguments
   - Log invalid calls for prompt improvement
   - Example: "Location not found. Try 'City, Country' format"

4. **Timeout Scenarios**
   - Set reasonable timeouts (5-30s per function)
   - Kill long-running functions
   - Return partial results if possible
   - Fallback to simpler alternatives
   - User message: "Taking longer than expected..."

5. **Function Execution Errors**
   - Try-catch around all function calls
   - Return structured error responses
   - Distinguish user errors from system errors
   - Log with context for debugging
   - Suggest alternative approaches

**Implementation Example:**
\`\`\`python
async def execute_with_resilience (func_name, args):
    for attempt in range(3):
        try:
            result = await execute_function (func_name, args, timeout=30)
            return {"success": True, "data": result}
        except RateLimitError as e:
            await asyncio.sleep (parse_retry_after (e.headers))
        except TimeoutError:
            if attempt == 2:
                return {"success": False, "error": "Timeout", "suggestion": "Try simpler query"}
            await asyncio.sleep(2 ** attempt)
        except Exception as e:
            return {"success": False, "error": str (e), "type": type (e).__name__}
\`\`\`

**User Experience:**
- Show progress indicators for long operations
- Provide actionable error messages
- Offer alternatives when primary fails
- Never expose raw stack traces
- Track error patterns for improvement

**Monitoring:**
- Alert on error rate > 5%
- Track error types and frequency
- Measure retry success rates
- Monitor timeout patterns
- Dashboard for real-time status`,
    keyPoints: [
      'Implement exponential backoff with jitter for network errors',
      'Use circuit breakers and rate limiting strategies',
      'Provide clear error messages and fallback options',
      'Monitor error patterns and success rates',
    ],
  },
  {
    id: 'q3',
    question:
      'Compare the function calling implementations across different LLM providers (OpenAI, Anthropic Claude, Google Gemini). What are the key differences in their approaches, and how would you design a provider-agnostic function calling system?',
    sampleAnswer: `Different LLM providers have varying approaches to function calling, each with trade-offs:

**OpenAI (GPT-4/GPT-3.5-Turbo):**
- Uses "functions" parameter in API
- Supports parallel function calling (GPT-4 Turbo)
- function_call parameter: "auto", "none", or specific function
- Returns function_call in message
- JSON mode for structured outputs
- Pros: Mature, well-documented, reliable
- Cons: No streaming for function calls initially

**Anthropic Claude:**
- Uses "tools" terminology instead of "functions"
- tool_use message type
- Streaming support for tool calls
- Computer use capability (controlling UI)
- input_schema instead of parameters
- Pros: Strong reasoning, clear documentation
- Cons: Newer, smaller ecosystem

**Google Gemini:**
- function_declarations parameter
- functionCall response type
- Part-based message structure
- Native multi-modal tool use
- Pros: Integrated with Google services
- Cons: Different message format, less mature

**Key Differences:**1. **Schema Format:**
   - OpenAI: "parameters" with JSON Schema
   - Claude: "input_schema" with JSON Schema
   - Gemini: "parameters" in FunctionDeclaration

2. **Response Format:**
   - OpenAI: message.function_call
   - Claude: content blocks with tool_use type
   - Gemini: candidates[].content.parts[].functionCall

3. **Parallel Calls:**
   - OpenAI GPT-4 Turbo: Native support
   - Claude: Sequential by default
   - Gemini: Supports multiple in parts

**Provider-Agnostic Design:**

\`\`\`python
class UniversalFunctionCaller:
    """Provider-agnostic function calling."""
    
    def __init__(self, provider: str):
        self.provider = provider
        self.adapter = self._get_adapter (provider)
    
    def call (self, messages, functions):
        # Convert to provider format
        provider_messages = self.adapter.convert_messages (messages)
        provider_functions = self.adapter.convert_functions (functions)
        
        # Call provider
        response = self.adapter.call_api (provider_messages, provider_functions)
        
        # Convert response to universal format
        return self.adapter.parse_response (response)
    
    class OpenAIAdapter:
        def convert_functions (self, functions):
            return [{"name": f.name, "parameters": f.schema} for f in functions]
        
        def parse_response (self, response):
            if response.message.function_call:
                return {
                    "type": "function_call",
                    "name": response.message.function_call.name,
                    "arguments": json.loads (response.message.function_call.arguments)
                }
    
    class ClaudeAdapter:
        def convert_functions (self, functions):
            return [{"name": f.name, "input_schema": f.schema} for f in functions]
        
        def parse_response (self, response):
            for block in response.content:
                if block.type == "tool_use":
                    return {
                        "type": "function_call",
                        "name": block.name,
                        "arguments": block.input
                    }
\`\`\`

**Best Practices:**1. Abstract function schemas into universal format
2. Use adapters for provider-specific conversions
3. Handle provider-specific features gracefully
4. Test with multiple providers
5. Fallback to alternative provider on failure
6. Document provider-specific quirks
7. Monitor provider reliability

**Migration Strategy:**
- Start with one provider
- Add abstraction layer early
- Test new providers in parallel
- Gradual migration with feature flags
- Keep provider-specific optimizations optional`,
    keyPoints: [
      'Different providers have varying function calling implementations',
      'Use adapter pattern for provider-agnostic design',
      'Abstract function schemas into universal format',
      'Implement graceful fallbacks between providers',
    ],
  },
];
