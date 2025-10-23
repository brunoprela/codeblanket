/**
 * Quiz questions for Prompt Engineering for Code section
 */

export const promptengineeringforcodeQuiz = [
  {
    id: 'bcgs-prompteng-q-1',
    question:
      'Explain the "context sandwich" pattern for code generation prompts. Why is this structure effective, and what are the consequences of getting the order wrong?',
    hint: 'Think about how LLMs process information and make decisions.',
    sampleAnswer:
      'The context sandwich pattern structures prompts as: Context → Request → Constraints. This is effective because: 1) **Context First** - Establishes the environment (file structure, imports, existing code) before asking for changes. LLMs use this to ground their response. 2) **Request in Middle** - The actual task, now informed by context. The LLM knows what already exists and can generate complementary code. 3) **Constraints Last** - Specific requirements (type hints, error handling, etc.) that refine the generation. These stick in the LLM\'s "mind" as it generates. Getting order wrong has consequences: **Request First** - LLM generates generic code without context, likely incompatible with project. **Constraints First** - LLM may forget constraints by the time it generates (recency bias). **Context Last** - LLM generates code, then realizes it conflicts with context, resulting in inconsistencies. Real example: Asking to "add error handling" without context produces generic try-catch. With context showing existing error patterns, it matches project style.',
    keyPoints: [
      'Context → Request → Constraints order leverages LLM information processing',
      'Context grounds generation in existing codebase',
      'Constraints placed last benefit from recency bias',
      'Wrong order produces generic or incompatible code',
    ],
  },
  {
    id: 'bcgs-prompteng-q-2',
    question:
      'You need to provide file context to an LLM for code editing, but the file is 2000 lines and would consume most of your context window. Design a smart truncation strategy that preserves the most relevant information.',
    hint: 'Consider what information is most critical for different types of edits.',
    sampleAnswer:
      'Smart truncation strategy for large files: 1) **Identify Edit Location** - Know which line/function is being edited. 2) **Context Window Approach** - Keep ~50 lines before and after edit location (where most relevant context exists). 3) **Preserve Key Sections** - Even outside the window, keep: a) All imports (critical for dependencies), b) Class/function signatures (understanding structure), c) Type definitions (for type correctness). 4) **Summarize Middle Sections** - For functions between kept sections, replace bodies with comments: "def process_data(): # [50 lines - processes user input]". 5) **Smart Selection by Edit Type**: For adding error handling, keep error patterns from elsewhere in file; For type hints, keep existing type patterns; For new function, keep similar functions. 6) **Token Budget Allocation**: Imports (10%), Edit context (60%), Signatures (20%), Summaries (10%). Example: 2000-line file, editing line 1250 → Keep lines 1200-1300 + imports + all function signatures + summarize rest. This preserves most critical info in ~300 lines vs 2000.',
    keyPoints: [
      'Use context window around edit location (~50 lines each side)',
      'Always preserve imports and function/class signatures',
      'Summarize function bodies outside edit area',
      'Adapt strategy based on edit type (error handling, types, etc.)',
    ],
  },
  {
    id: 'bcgs-prompteng-q-3',
    question:
      'How would you design a prompt for generating a new function that needs to interact with multiple existing functions in the codebase? What information is essential vs. nice-to-have?',
    hint: 'Consider dependencies, types, style patterns, and examples.',
    sampleAnswer:
      '**Essential Information (Must Include):** 1) **Function Signatures** of all functions the new function will call - parameters, return types, what they do. Without this, LLM might hallucinate interfaces. 2) **Type Definitions** used in signatures - User, Response, etc. Prevents type mismatches. 3) **Import Statements** - Shows what libraries are available. 4) **Error Handling Patterns** - How existing code handles errors (exceptions, Result types, etc.). 5) **The Request** - Clear description of what new function should do. **Nice-to-Have (Include if space):** 1) **Example Usage** of similar existing functions - Shows patterns. 2) **Docstring Style** examples - Maintains consistency. 3) **Test Examples** from existing functions - Shows expected patterns. 4) **Project Conventions** document if exists. **Prompt Structure:** Context: [Existing function signatures + types + imports], Request: [Create new function that...], Constraints: [Match error handling patterns, use types consistently, follow same doc style], Examples: [Similar function from codebase]. Real example: Creating "process_user_payment()" that calls "validate_user()", "charge_card()", "send_receipt()" - must include all three signatures, User and Payment types, and error patterns.',
    keyPoints: [
      'Essential: function signatures, types, imports, error patterns',
      'Nice-to-have: examples, docstring styles, test patterns',
      'Must prevent interface hallucination',
      'Show patterns to maintain consistency',
    ],
  },
];
