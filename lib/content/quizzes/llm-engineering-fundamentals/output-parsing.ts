/**
 * Quiz questions for Output Parsing & Structured Data section
 */

export const outputparsingQuiz = [
  {
    id: 'q1',
    question:
      'Explain the advantages of using JSON mode with Pydantic models over regex parsing for extracting structured data from LLM outputs. Provide a concrete example where regex would fail.',
    sampleAnswer:
      'JSON mode with Pydantic is vastly superior because: (1) Guaranteed valid JSON - regex parsing can fail on escaped quotes, nested structures, or malformed output. JSON mode forces LLM to return parseable JSON. (2) Type validation - Pydantic automatically validates types (int vs string), email format, date format, ranges. Regex only matches patterns, not semantic correctness. (3) Nested structures - handling nested objects/arrays with regex is nightmare. JSON handles naturally. (4) Error messages - Pydantic gives clear validation errors. Regex gives cryptic "no match" or wrong data. (5) Schema evolution - adding fields to Pydantic model is trivial, regex patterns must be rewritten. Concrete failure example: Extracting person data. LLM returns: "The person is John O\'Brien (john.o\'brien+work@email-domain.com), age 30". Regex approach: email_pattern = r\'(\\\\S+@\\\\S+)\' might capture \'brien+work@email-domain.com\' (wrong!) or fail on apostrophe in name. Age regex r\'age (\\\\d+)\' works but if LLM says "30 years old" vs "age 30", regex fails. Phone numbers with different formats? Nightmare. JSON+Pydantic approach: class Person(BaseModel): name: str, email: EmailStr (validates format!), age: int. LLM returns: {"name": "John O\'Brien", "email": "john.o\'brien+work@email-domain.com", "age": 30}. Pydantic parses and validates - EmailStr validates email is valid, age is int not string. If LLM returns {"age": "30"}, Pydantic converts string to int automatically. If email is invalid, clear error. Production impact: Regex-based extraction has 10-30% failure rate due to format variations. JSON+Pydantic has <1% failure rate (only when LLM doesn\'t follow JSON despite mode). Time saved in debugging and handling edge cases is enormous.',
    keyPoints: [
      'JSON mode guarantees parseable output',
      'Pydantic validates types and formats automatically',
      'Regex fails on format variations and nested data',
      'Clear error messages vs regex matching failures',
      'Production failure rates: 10-30% regex vs <1% JSON+Pydantic',
    ],
  },
  {
    id: 'q2',
    question:
      "You're using JSON mode to extract data, but occasionally the LLM returns invalid JSON despite the mode being enabled. Describe a robust strategy for handling and recovering from these failures.",
    sampleAnswer:
      'Multi-layer failure handling strategy: Layer 1 - Detection: Try parsing JSON immediately, catch JSONDecodeError, log failed attempts with full output for debugging, track failure rate metrics. Layer 2 - Common fixes: (1) Extract JSON from markdown - LLM might wrap in ```json ... ```, use regex to extract content between backticks. (2) Find JSON block - search for first { to last }, sometimes LLM adds text before/after JSON. (3) Fix common mistakes - replace single quotes with double quotes, add missing closing braces if obvious, remove trailing commas. Layer 3 - LLM self-correction: Send response back to LLM with error, ask it to fix the JSON, include the parse error message, works for ~80% of failures. Example: messages.append({"role": "assistant", "content": bad_json}), messages.append({"role": "user", "content": f"That JSON is invalid: {error}. Please provide valid JSON."}). Layer 4 - Schema validation retry: If JSON parses but Pydantic validation fails, tell LLM which fields are missing/invalid, request completion with specific guidance. Layer 5 - Fallback extraction: If all else fails, use regex for critical fields, return partial data with is_complete=False flag, alert monitoring system. Implementation: max_retries = 3, try parse → if fail, try extraction tricks → if still fail, ask LLM to fix → if still fail, attempt partial extraction → if still fail, return error with original output saved. Prevention: (1) Use temperature=0 for structured extraction (more reliable), (2) Clear schema in system prompt, (3) Few-shot examples of correct JSON, (4) Validate with smaller model first in testing. Monitoring: Track failure rates, alert if >5%, analyze failed outputs to improve prompts. Production result: With this strategy, 99%+ requests succeed even when initial JSON fails.',
    keyPoints: [
      'Multi-layer strategy from simple fixes to LLM retry',
      'Extract JSON from markdown or surrounding text',
      'Ask LLM to self-correct with error message',
      'Fallback to partial extraction if needed',
      'Monitor failure rates and improve prompts',
    ],
  },
  {
    id: 'q3',
    question:
      'Compare the Instructor library versus manually using JSON mode with Pydantic. When would you choose each approach, and what are the trade-offs?',
    sampleAnswer:
      "Instructor (high-level abstraction): Advantages: (1) One-line extraction - client.create(response_model=MyModel) handles everything, no manual JSON parsing, (2) Automatic retries - if validation fails, Instructor retries with feedback automatically, (3) Streaming support - can stream structured data, (4) Less boilerplate - no manual message construction, (5) Retry strategies built-in. Trade-offs: (1) Another dependency to manage, (2) Less control over retry logic, (3) Abstracts away details (good and bad), (4) Tied to Instructor updates/changes, (5) Slightly slower due to abstraction. Manual JSON mode with Pydantic: Advantages: (1) Full control over prompt and retry logic, (2) No external dependencies beyond Pydantic, (3) Explicit and clear what's happening, (4) Can customize every aspect, (5) Easier to debug when issues occur. Trade-offs: (1) More boilerplate code, (2) Must implement retry logic yourself, (3) More code to maintain, (4) Easy to make mistakes in error handling. When to choose Instructor: (1) Rapid development - building MVP or prototype, (2) Standard use cases - simple extraction without special logic, (3) Team prefers high-level APIs, (4) Want built-in best practices, (5) Many extraction use cases (leverage library features). When to choose manual: (1) Complex custom logic - special retry strategies, multi-step validation, (2) Minimal dependencies preferred, (3) Performance critical (eliminate abstraction overhead), (4) Need full control and transparency, (5) Building own framework/library. Hybrid approach (recommended): Use Instructor for 80% of standard cases, use manual for 20% that need custom control, wrap both in your own abstraction so switching is easy. Personal preference: Start with Instructor to validate approach quickly, if you hit limitations or need custom behavior, migrate to manual. For production systems handling millions of requests, I prefer manual for predictability and control, but Instructor is excellent for getting started quickly.",
    keyPoints: [
      'Instructor simplifies with one-line extraction',
      'Manual approach provides full control',
      'Instructor better for rapid development',
      'Manual better for custom logic and minimal dependencies',
      'Hybrid approach recommended for flexibility',
    ],
  },
];
