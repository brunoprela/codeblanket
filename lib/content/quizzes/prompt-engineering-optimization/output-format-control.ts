/**
 * Quiz questions for Output Format Control section
 */

export const outputformatcontrolQuiz = [
    {
        id: 'peo-format-q-1',
        question:
            'Why is JSON the preferred format for structured LLM outputs? Compare JSON to alternatives like XML and plain text in production contexts.',
        hint: 'Consider parseability, validation, type safety, and ecosystem support.',
        sampleAnswer:
            'JSON preferred because: 1) PARSEABILITY: Built-in parser in all languages, handles nested structures naturally, clear syntax reduces parsing errors; 2) VALIDATION: JSON Schema provides rigorous validation, catch errors before processing; 3) TYPE SAFETY: Use Pydantic/TypeScript for compile-time types, runtime validation; 4) ECOSYSTEM: Every tool/library supports JSON, easy integration; 5) CONSISTENCY: LLMs trained on JSON, produce it reliably. COMPARED TO XML: JSON simpler, less verbose, easier to parse, but XML better for mixed content (text + structure). COMPARED TO PLAIN TEXT: JSON structured and parseable, plain text requires custom parsing, regex, prone to format variations. IN PRODUCTION: JSON enables reliable automation - parse once, validate schema, type-safe access. Alternatives used when: XML for documents with narrative + structure; plain text when simplicity critical and format varies. Instructor library leverages JSON for type-safe LLM outputs.',
        keyPoints: [
            'JSON easiest to parse in all languages',
            'JSON Schema enables rigorous validation',
            'Works with type systems (Pydantic, TypeScript)',
            'LLMs produce JSON most reliably',
            'Simpler and less error-prone than XML',
            'Essential for reliable automation',
        ],
    },
    {
        id: 'peo-format-q-2',
        question:
            'Explain the Instructor library approach to structured outputs. What problems does it solve compared to manual JSON parsing?',
        hint: 'Think about type safety, validation, error handling, and developer experience.',
        sampleAnswer:
            'Instructor wraps OpenAI API with Pydantic models for type-safe outputs. APPROACH: Define Pydantic model describing output schema → pass as response_model → Instructor handles prompting and validation → get typed object, not string. PROBLEMS SOLVED: 1) TYPE SAFETY: Autocomplete, compile-time checking, runtime validation; 2) VALIDATION: Automatic constraint checking (ranges, required fields, formats); 3) ERROR HANDLING: Clear errors if model produces invalid format, automatic retry with feedback; 4) PARSING: No manual JSON.loads() + validation logic; 5) MAINTAINABILITY: Schema as code, changes type-checked. WITHOUT INSTRUCTOR: Manual JSON parsing, check every field exists, validate types, handle errors, no autocomplete. WITH INSTRUCTOR: Define schema once, get guaranteed valid objects, focus on business logic. EXAMPLE: response_model=User → get User object with guaranteed name:str, age:int, email:str fields. Production benefit: Catch format errors before they cause runtime failures, reduce boilerplate 10x, self-documenting schemas.',
        keyPoints: [
            'Wraps LLM API with Pydantic for type safety',
            'Automatic validation against schema',
            'Autocomplete and compile-time type checking',
            'Eliminates manual parsing and validation code',
            'Clear error messages on validation failure',
            'Reduces boilerplate and improves maintainability',
        ],
    },
    {
        id: 'peo-format-q-3',
        question:
            'How do you handle malformed LLM outputs in production? Design a robust retry strategy with format correction.',
        hint: 'Consider validation, feedback loops, maximum retries, and fallback strategies.',
        sampleAnswer:
            'Robust handling strategy: 1) VALIDATION: Parse output, validate against schema (JSON Schema, Pydantic), catch specific errors (missing field, wrong type, format violation); 2) SMART RETRY WITH FEEDBACK: On validation failure, send error back to LLM with correction guidance: "Previous output failed: field X missing, expected type Y. Fix and retry." Include original prompt + error details; 3) MAX RETRIES: Limit to 3 attempts (diminishing returns, cost control); 4) FALLBACK: After max retries, either return structured error to user, use best-effort parsing (extract what\'s valid), or route to human review queue; 5) LEARNING: Log failures for prompt improvement, identify patterns. IMPLEMENTATION: try-except around parse, validation function with detailed errors, retry loop with backoff, error categorization (recoverable vs not), metrics on retry rates. PRODUCTION EXAMPLE: 1st attempt fails → retry with "field missing" feedback → 2nd succeeds 80% of time → 3rd attempt catches most remaining → <1% to human review. Reduces error rate from 10% to <1%.',
        keyPoints: [
            'Validate immediately after getting output',
            'Retry with specific error feedback to LLM',
            'Limit retries (typically 3 maximum)',
            'Have fallback strategy for persistent failures',
            'Log and analyze failure patterns',
            'Route unrecoverable errors to human review',
        ],
    },
];

