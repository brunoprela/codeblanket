export const structuredToolResponsesQuiz = [
  {
    id: 'q1',
    question:
      'Design a standardized response format for all tools that maximizes LLM understanding while maintaining flexibility for different tool types. How would you handle success, partial success, and error cases?',
    sampleAnswer: `A standardized response format should have consistent top-level structure across all tools with status indicators, data payloads, error information, and metadata. Use a three-state model: success (operation completed), partial (some parts succeeded), and error (operation failed). Include clear error messages with suggestions for recovery, structured data in consistent format, metadata like timestamps and costs, and natural language summaries for easy LLM consumption. The format should be self-describing with context about what each field means.`,
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
      'Explain how you would design tool responses for complex, nested data structures while keeping them understandable for LLMs. Provide examples of good and bad response designs.',
    sampleAnswer: `For complex data, flatten where possible but use nesting for logical grouping. Include descriptions in the response explaining what each section contains. Bad design: deeply nested objects without context, ambiguous field names, mixing data types. Good design: shallow hierarchy with clear sections, descriptive field names, consistent types, summary fields for key information, units specified for all measurements, examples embedded in responses. Always include a natural language summary at the top level for easy LLM comprehension.`,
    keyPoints: [
      'Key concept from discussion question',
      'Key concept from discussion question',
      'Key concept from discussion question',
      'Key concept from discussion question',
    ],
  },
  {
    id: 'q3',
    question:
      'How would you implement response validation to ensure tools always return properly formatted data? What would you do when validation fails in production?',
    sampleAnswer: `Use Pydantic models or JSON Schema to define expected response structure. Validate all tool responses before returning to LLM. On validation failure, log the error with full context, return a properly formatted error response instead of invalid data, alert developers about the issue, and implement graceful degradation. Have fallback responses for common validation failures. Monitor validation failure rates and fix tools that frequently produce invalid responses. In development, fail loudly; in production, handle gracefully while alerting.`,
    keyPoints: [
      'Key concept from discussion question',
      'Key concept from discussion question',
      'Key concept from discussion question',
      'Key concept from discussion question',
    ],
  },
];
