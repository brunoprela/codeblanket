export const toolUsePromptingQuiz = [
  {
    id: 'q1',
    question:
      'Design a comprehensive system prompt that teaches an LLM when and how to use tools effectively. What guidelines, examples, and constraints would you include, and how would you adapt the prompt based on the specific tools available?',
    sampleAnswer: `An effective tool-use system prompt should be clear, structured, and context-aware. Key components include explicit guidelines for tool selection, usage examples, and situational awareness.

The prompt should start with role definition, explain available tools with clear descriptions, provide decision-making guidelines (when to use tools vs answer directly), include concrete examples of good tool usage, and set constraints on tool use patterns.

Best practices include keeping the prompt concise yet comprehensive, using bullet points for readability, providing examples of both tool use and direct responses, explaining error handling, and adapting based on available tools and user context.`,
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
      'Explain how you would debug and improve prompts when LLMs consistently make poor tool choices or provide incorrect arguments. What testing methodology would you use?',
    sampleAnswer: `Debugging tool-use prompts requires systematic testing and iteration. Start by collecting failure cases where the LLM chose wrong tools or provided bad arguments.

Analyze patterns in failures - is it specific tools, argument types, or query phrasings? Create a test suite with diverse scenarios covering common cases, edge cases, and known failure modes.

A/B test prompt variations, measuring success rates for tool selection accuracy and argument correctness. Use few-shot examples to guide the LLM toward better behavior for problematic cases.

Implement monitoring in production to track tool selection accuracy, argument validation failures, and user corrections. Use this data to continuously improve prompts through iterative refinement.`,
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
      'Compare different prompting strategies for tool use: zero-shot, few-shot, and chain-of-thought. When would you use each approach, and how do they affect reliability and cost?',
    sampleAnswer: `Each prompting strategy has distinct trade-offs:

Zero-shot relies solely on tool descriptions and system instructions. It's token-efficient and works well with clear tool schemas, but may struggle with ambiguous cases or complex decision-making.

Few-shot includes concrete examples of correct tool usage, dramatically improving accuracy for similar queries. However, it increases token usage and requires careful example selection.

Chain-of-thought (CoT) prompts the LLM to reason explicitly before choosing tools, improving complex decision-making but adding latency and token cost.

Best practice is hybrid: use zero-shot as baseline, add few-shot examples for problematic tools or patterns, and enable CoT for complex multi-step tasks. Cost scales with context length, so balance accuracy needs against budget constraints.`,
    keyPoints: [
      'Key concept from discussion question',
      'Key concept from discussion question',
      'Key concept from discussion question',
      'Key concept from discussion question',
    ],
  },
];
