/**
 * Multiple choice questions for Output Validation & Guardrails section
 */

export const outputvalidationguardrailsMultipleChoice = [
  {
    id: 'output-val-mc-1',
    question:
      'Your LLM should output JSON: {"name": str, "age": int}. It outputs {"name": "John", "age": "30", "city": "NYC"}. What validation errors exist?',
    options: [
      'Only missing fields',
      'Only extra fields (city)',
      'Only wrong type (age is string not int)',
      'Wrong type (age) AND extra field (city)',
    ],
    correctAnswer: 3,
    explanation:
      'Two validation errors: (1) age is string "30" not integer 30 (wrong type), (2) city field should not exist (extra field). Both need fixing. Options A, B, C only identify one error each.',
  },
  {
    id: 'output-val-mc-2',
    question:
      'Your output validation system rejects 15% of LLM outputs. What is the PRIMARY concern?',
    options: [
      'LLM is wasting resources generating invalid outputs',
      'Validation logic is too strict',
      'Need to improve prompts to reduce failures',
      'Users experience 15% request failures',
    ],
    correctAnswer: 3,
    explanation:
      'The primary concern is user experience—15% of user requests fail, which is unacceptable. While options A and C are true (should be addressed), the immediate concern is users seeing failures. Option B might be true if false positives are high, but we should first improve generation quality.',
  },
  {
    id: 'output-val-mc-3',
    question:
      'When output validation fails, what is the BEST immediate action?',
    options: [
      'Return error to user immediately',
      'Retry generation with validation feedback',
      'Return a hardcoded safe default',
      'Log error and continue without output',
    ],
    correctAnswer: 1,
    explanation:
      'Retry with validation feedback gives the LLM a chance to correct its error. Include the validation error in the retry prompt: "Previous output had wrong type for age field. Generate valid JSON." This often succeeds. Option A (immediate error) doesn\'t attempt recovery. Option C (default) loses user\'s specific request.',
  },
  {
    id: 'output-val-mc-4',
    question:
      'You use Pydantic for output validation. An output has age=150 (valid int, but unrealistic). What type of validation is needed?',
    options: [
      'Schema validation (Pydantic handles this)',
      'Business logic validation (age range 0-120)',
      'Type validation (int vs string)',
      'Format validation (JSON vs XML)',
    ],
    correctAnswer: 1,
    explanation:
      'This requires business logic validation—age must be reasonable (0-120). Pydantic handles schema/type validation (A, C). Format validation (D) is earlier in the pipeline. You need custom validation: age: int = Field(..., ge=0, le=120).',
  },
  {
    id: 'output-val-mc-5',
    question:
      'Your validation adds 200ms latency. Which optimization gives the LARGEST latency reduction?',
    options: [
      'Run validations in parallel instead of serially',
      'Cache validation results for similar outputs',
      'Use faster validation libraries',
      'Skip validation for trusted users',
    ],
    correctAnswer: 1,
    explanation:
      'Caching gives the largest reduction if you have repeated/similar outputs (common in many apps). 80% cache hit rate = 80% of requests complete in ~2ms vs 200ms. Option A (parallel) helps but limited by slowest check. Option D (skip validation) is dangerous—never skip safety checks.',
  },
];
