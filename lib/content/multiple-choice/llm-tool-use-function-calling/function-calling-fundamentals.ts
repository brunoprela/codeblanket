import { MultipleChoiceQuestion } from '../../../types';

export const functionCallingFundamentalsMultipleChoice: MultipleChoiceQuestion[] =
  [
    {
      id: 'fcfc-mc-1',
      question:
        'What is the primary advantage of function calling over parsing text responses from an LLM?',
      options: [
        'Function calling is faster',
        'Function calling provides structured, validated output with guaranteed format',
        'Function calling uses fewer tokens',
        'Function calling works with older models',
      ],
      correctAnswer: 1,
      explanation:
        'Function calling provides structured output with schema validation, ensuring consistent format and type safety. This is much more reliable than parsing potentially inconsistent text responses.',
    },
    {
      id: 'fcfc-mc-2',
      question:
        'In OpenAI\'s function calling API, what does the "function_call" parameter value "auto" mean?',
      options: [
        'The model will always call a function',
        'The model decides whether to call a function based on context',
        'Functions are called automatically in the background',
        'The model will never call functions',
      ],
      correctAnswer: 1,
      explanation:
        'With function_call="auto", the model analyzes the user query and decides whether calling a function is necessary, or if it can respond directly from its knowledge.',
    },
    {
      id: 'fcfc-mc-3',
      question:
        'What is the correct order of operations in a function calling workflow?',
      options: [
        'Execute function → Call LLM → Parse response → Return to user',
        'Call LLM → Execute function → Add result to messages → Call LLM again → Return response',
        'Parse user input → Execute function → Call LLM → Return response',
        'Call LLM → Return function name to user → Wait for execution → Call LLM again',
      ],
      correctAnswer: 1,
      explanation:
        'The workflow is: 1) Call LLM with functions parameter, 2) LLM generates function call, 3) Your code executes the function, 4) Add function result to messages, 5) Call LLM again to synthesize the response.',
    },
    {
      id: 'fcfc-mc-4',
      question:
        'Which HTTP status code indicates rate limiting that requires exponential backoff?',
      options: [
        '400 Bad Request',
        '403 Forbidden',
        '429 Too Many Requests',
        '503 Service Unavailable',
      ],
      correctAnswer: 2,
      explanation:
        '429 Too Many Requests indicates rate limiting. Implement exponential backoff and respect the Retry-After header to handle this gracefully.',
    },
    {
      id: 'fcfc-mc-5',
      question:
        "What happens if an LLM generates a function_call but the function doesn't exist in your registry?",
      options: [
        'The LLM automatically retries with a different function',
        'Your application should handle this as an error and inform the LLM',
        'The OpenAI API throws an error automatically',
        'The function_call is silently ignored',
      ],
      correctAnswer: 1,
      explanation:
        'Your application must handle missing functions by returning an error response to the LLM, which can then try a different approach or ask for clarification.',
    },
  ];
